import os
import numpy as np
import scipy as sp
import dimod
import minorminer
import gurobipy
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import LeapHybridSampler
from dwave.system import DWaveSampler, FixedEmbeddingComposite


class QUnfoldQUBO:
    def __init__(self, response, measured, lam=0.0):
        self.R = response
        self.d = measured
        self.lam = lam

        dim = len(measured)
        main_d = np.array([0, -1] + [-2] * (dim - 4) + [-1, 0])
        sup_sub_d = np.array([0] + [1] * (dim - 3) + [0])
        self.D = np.diag(main_d) + np.diag(sup_sub_d, k=1) + np.diag(sup_sub_d, k=-1)

    @property
    def num_bins(self):
        return len(self.d)

    @property
    def num_logical_qubits(self):
        return self.dwave_bqm.num_variables

    @property
    def num_physical_qubits(self):
        return sum(len(chain) for chain in self.graph_embedding.values())

    @property
    def num_bits(self):
        eff = np.sum(self.R, axis=0)
        eff[np.isclose(eff, 0)] = 1
        exp = np.ceil(self.d / eff)
        return [int(np.ceil(np.log2(x * 1.2))) if x else 1 for x in exp]

    @property
    def precision_vectors(self):
        num_bits = self.num_bits
        pvecs = [2 ** np.arange(num_bits[i]) for i in range(self.num_bins)]
        return pvecs

    @property
    def linear_coeffs(self):
        return -2 * (self.R.T @ self.d)

    @property
    def quadratic_coeffs(self):
        return (self.R.T @ self.R) + self.lam * (self.D.T @ self.D)

    def _get_qubo_matrix(self):
        dim = self.num_bins
        a = self.linear_coeffs
        B = self.quadratic_coeffs
        pvecs = self.precision_vectors
        linear_blocks = [a[i] * np.diag(pvecs[i]) for i in range(dim)]
        quadratic_blocks = [
            [B[i, j] * np.outer(pvecs[i], pvecs[j]) for j in range(dim)]
            for i in range(dim)
        ]
        return sp.linalg.block_diag(*linear_blocks) + np.block(quadratic_blocks)

    def _get_dwave_bqm(self):
        Q = self.qubo_matrix
        size = len(Q)
        linear = {i: Q[i, i] for i in range(size)}
        quadratic = {
            (i, j): 2 * Q[i, j] for i in range(size) for j in range(i + 1, size)
        }
        return dimod.BinaryQuadraticModel(linear, quadratic, vartype=dimod.BINARY)

    def _post_process_sampleset(self, sampleset):
        pvecs = self.precision_vectors
        indices = np.cumsum([0] + [len(pvec) for pvec in pvecs[:-1]])
        solutions = [
            np.add.reduceat(rec.sample * np.concatenate(pvecs), indices=indices)
            for rec in sampleset.record
        ]
        energies = [rec.energy for rec in sampleset.record]
        min_energy = min(energies)
        beta_boltzmann = 100
        weights = np.exp(-beta_boltzmann * np.abs(np.array(energies) - min_energy))
        return np.average(solutions, weights=weights, axis=0)

    def _get_graph_embedding(self, **kwargs):
        source_edgelist = list(self.dwave_bqm.quadratic) + list(
            (v, v) for v in self.dwave_bqm.linear
        )
        target_edgelist = self._sampler.edgelist
        return minorminer.find_embedding(S=source_edgelist, T=target_edgelist, **kwargs)

    def _run_montecarlo_toys(self, num_toys, num_cores, **kwargs):
        def run_toy(toy):
            toy.initialize_qubo_model()
            if isinstance(self._sampler, DWaveSampler):
                embedding = toy._get_graph_embedding()
                sampler = FixedEmbeddingComposite(self._sampler, embedding=embedding)
            else:
                sampler = self._sampler
            sampleset = sampler.sample(toy.dwave_bqm, **kwargs)
            solution = toy._post_process_sampleset(sampleset)
            return solution

        smeared_d = np.random.poisson(self.d, size=(num_toys, self.num_bins))
        toys_gen = (QUnfoldQUBO(self.R, d, self.lam) for d in smeared_d)
        if num_cores is None:
            num_cores = os.cpu_count() - 2
        desc = "Running MC toys"
        with ThreadPoolExecutor(num_cores) as executor:
            res = list(tqdm(executor.map(run_toy, toys_gen), total=num_toys, desc=desc))
        cov = np.cov(res, rowvar=False)
        err = np.sqrt(np.diag(cov))
        return err, cov

    def initialize_qubo_model(self):
        self.qubo_matrix = self._get_qubo_matrix()
        self.dwave_bqm = self._get_dwave_bqm()

    def solve_simulated_annealing(
        self, num_reads, num_toys=None, num_cores=None, seed=None
    ):
        self._sampler = SimulatedAnnealingSampler()
        sampleset = self._sampler.sample(self.dwave_bqm, num_reads=num_reads, seed=seed)
        sol = self._post_process_sampleset(sampleset)
        if num_toys is None:
            err = np.sqrt(sol)
            cov = np.diag(sol)
        else:
            err, cov = self._run_montecarlo_toys(
                num_toys, num_cores, num_reads=num_reads, seed=seed
            )
        return sol, err, cov

    def solve_hybrid_sampler(self, num_toys=None, num_cores=None):
        self._sampler = LeapHybridSampler()
        sampleset = self._sampler.sample(self.dwave_bqm)
        sol = self._post_process_sampleset(sampleset)
        if num_toys is None:
            err = np.sqrt(sol)
            cov = np.diag(sol)
        else:
            err, cov = self._run_montecarlo_toys(num_toys, num_cores)
        return sol, err, cov

    def set_quantum_device(self, device_name=None, dwave_token=None):
        self._sampler = DWaveSampler(solver=device_name, token=dwave_token)

    def set_graph_embedding(self, **kwargs):
        self.graph_embedding = self._get_graph_embedding(**kwargs)

    def solve_quantum_annealing(self, num_reads, num_toys=None, num_cores=None):
        sampler = FixedEmbeddingComposite(self._sampler, embedding=self.graph_embedding)
        sampleset = sampler.sample(self.dwave_bqm, num_reads=num_reads)
        sol = self._post_process_sampleset(sampleset)
        if num_toys is None:
            err = np.sqrt(sol)
            cov = np.diag(sol)
        else:
            err, cov = self._run_montecarlo_toys(
                num_toys, num_cores, num_reads=num_reads
            )
        return sol, err, cov

    def compute_energy(self, x):
        xbin = []
        num_bits = self.num_bits
        for i in range(self.num_bins):
            bitstr = bin(int(x[i]))[2:].zfill(num_bits[i])
            xbin.extend(map(int, bitstr[::-1]))
        energy = self.dwave_bqm.energy(sample=xbin)
        return energy

    def solve_gurobi_integer(self):
        model = gurobipy.Model()
        model.setParam("OutputFlag", 0)
        x = [
            model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=2**b - 1)
            for b in self.num_bits
        ]
        a = self.linear_coeffs
        B = self.quadratic_coeffs
        model.setObjective(a @ x + x @ B @ x, sense=gurobipy.GRB.MINIMIZE)
        model.optimize()
        sol = np.array([var.x for var in x])
        err = np.sqrt(sol)
        cov = np.diag(sol)
        return sol, err, cov

    def solve_gurobi_binary(self):
        model = gurobipy.Model()
        model.setParam("OutputFlag", 0)
        num_bits = self.num_bits
        x = [
            model.addVar(vtype=gurobipy.GRB.BINARY)
            for i in range(self.num_bins)
            for _ in range(num_bits[i])
        ]
        Q = self.qubo_matrix
        model.setObjective(x @ Q @ x, sense=gurobipy.GRB.MINIMIZE)
        model.optimize()
        bitstr = np.array([var.x for var in x], dtype=int)
        arrays = np.split(bitstr, np.cumsum(num_bits[:-1]))
        sol = np.array(
            [int("".join(arr.astype(str))[::-1], base=2) for arr in arrays], dtype=float
        )
        err = np.sqrt(sol)
        cov = np.diag(sol)
        return sol, err, cov
