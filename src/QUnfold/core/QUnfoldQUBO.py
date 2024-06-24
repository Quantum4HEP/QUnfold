import os
import tqdm
import gurobipy
import minorminer
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from pyqubo import LogEncInteger
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import LeapHybridSampler
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from functools import cached_property


class QUnfoldQUBO:

    def __init__(self, response, measured, lam=0.0):
        self.R = response
        self.d = measured
        self.lam = lam

    @staticmethod
    def _get_laplacian(dim):
        diag = np.array([-1] + [-2] * (dim - 2) + [-1])
        D = np.diag(diag).astype(float)
        diag1 = np.ones(dim - 1)
        D += np.diag(diag1, k=1) + np.diag(diag1, k=-1)
        return D

    @cached_property
    def num_bins(self):
        return len(self.d)

    @cached_property
    def num_logical_qubits(self):
        return self.dwave_bqm.num_variables

    @cached_property
    def num_physical_qubits(self):
        return sum(len(chain) for chain in self.graph_embedding.values())

    @cached_property
    def upper_bounds(self):
        efficiency = np.sum(self.R, axis=0)
        efficiency[efficiency == 0] = 1
        num_entries = self.d / efficiency
        return [
            int(2 ** np.ceil(np.log2((num_entries[bin] + 1) * 1.2))) - 1
            for bin in range(self.num_bins)
        ]

    @cached_property
    def pyqubo_variables(self):
        n = len(str(self.num_bins - 1))
        labels = ["x" + str(bin).zfill(n) for bin in range(self.num_bins)]
        bounds = self.upper_bounds
        return [
            LogEncInteger(label=labels[bin], value_range=(0, bounds[bin]))
            for bin in range(self.num_bins)
        ]

    @cached_property
    def linear_coeffs(self):
        return -2 * (self.R.T @ self.d)

    @cached_property
    def quadratic_coeffs(self):
        G = self._get_laplacian(dim=self.num_bins)
        return (self.R.T @ self.R) + self.lam * (G.T @ G)

    @cached_property
    def pyqybo_hamiltonian(self):
        x = self.pyqubo_variables
        a = self.linear_coeffs
        B = self.quadratic_coeffs
        hamiltonian = 0
        for i in range(self.num_bins):
            if a[i] != 0:
                hamiltonian += a[i] * x[i]
        for i in range(self.num_bins):
            for j in range(self.num_bins):
                if B[i, j] != 0:
                    hamiltonian += B[i, j] * x[i] * x[j]
        return hamiltonian

    @cached_property
    def pyqubo_model(self):
        return self.pyqybo_hamiltonian.compile()

    @cached_property
    def dwave_bqm(self):
        return self.pyqubo_model.to_bqm()

    @cached_property
    def qubo_matrix(self):
        vars = sorted(self.dwave_bqm.variables)
        qubo_dict, _ = self.dwave_bqm.to_qubo()
        Q = np.empty(shape=(len(vars), len(vars)))
        for i, u in enumerate(vars):
            for j, v in enumerate(vars):
                Q[i, j] = qubo_dict.get((u, v), 0)
        Q = 0.5 * (Q + Q.T)
        return Q

    @cached_property
    def graph_embedding(self):
        target_edgelist = self._sampler.edgelist
        source_edgelist = list(self.dwave_bqm.quadratic) + [
            (v, v) for v in self.dwave_bqm.linear
        ]
        return minorminer.find_embedding(S=source_edgelist, T=target_edgelist)

    def _post_process_sampleset(self, sampleset):
        all_samples = self.pyqubo_model.decode_sampleset(sampleset)
        solutions, energies = [], []
        for sample in all_samples:
            solutions.append([sample.subh[var.label] for var in self.pyqubo_variables])
            energies.append(sample.energy)
        min_energy = min(energies)
        weights = np.exp(-np.abs(np.array(energies) - min_energy) / abs(min_energy))
        solution = np.average(solutions, weights=weights, axis=0)
        return solution

    def _run_montecarlo_toys(self, num_toys, num_cores, **kwargs):
        def run_mc_toy(i):
            smeared_d = np.random.poisson(self.d)
            unfolder = QUnfoldQUBO(self.R, smeared_d, self.lam)
            if isinstance(self._sampler, DWaveSampler):
                sampler = FixedEmbeddingComposite(
                    self._sampler, embedding=unfolder.graph_embedding
                )
            else:
                sampler = self._sampler
            sampleset = sampler.sample(unfolder.dwave_bqm, **kwargs)
            solutions[i] = unfolder._post_process_sampleset(sampleset)

        if num_cores is None:
            num_cores = os.cpu_count() - 2
        solutions = np.empty(shape=(num_toys, self.num_bins))
        with ThreadPoolExecutor(num_cores) as executor:
            list(
                tqdm.tqdm(
                    executor.map(run_mc_toy, range(num_toys)),
                    total=num_toys,
                    desc="Running MC toys",
                )
            )
        cov = np.cov(solutions, rowvar=False)
        error = np.sqrt(np.diag(cov))
        corr = np.corrcoef(solutions, rowvar=False)
        return error, cov, corr

    def solve_simulated_annealing(
        self, num_reads, num_toys=None, num_cores=None, seed=None
    ):
        self._sampler = SimulatedAnnealingSampler()
        sampleset = self._sampler.sample(self.dwave_bqm, num_reads=num_reads, seed=seed)
        solution = self._post_process_sampleset(sampleset)
        if num_toys is None:
            error, cov, corr = np.sqrt(solution), None, None
        else:
            error, cov, corr = self._run_montecarlo_toys(
                num_toys, num_cores, num_reads=num_reads, seed=seed
            )
        return solution, error, cov, corr

    def solve_hybrid_sampler(self, num_toys=None, num_cores=None):
        self._sampler = LeapHybridSampler()
        sampleset = self._sampler.sample(self.dwave_bqm)
        solution = self._post_process_sampleset(sampleset)
        if num_toys is None:
            error, cov, corr = np.sqrt(solution), None, None
        else:
            error, cov, corr = self._run_montecarlo_toys(num_toys, num_cores)
        return solution, error, cov, corr

    def set_quantum_device(self, device_name):
        self._sampler = DWaveSampler(solver=device_name)

    def solve_quantum_annealing(self, num_reads, num_toys=None, num_cores=None):
        sampler = FixedEmbeddingComposite(self._sampler, embedding=self.graph_embedding)
        sampleset = sampler.sample(self.dwave_bqm, num_reads=num_reads)
        solution = self._post_process_sampleset(sampleset)
        if num_toys is None:
            error, cov, corr = np.sqrt(solution), None, None
        else:
            error, cov, corr = self._run_montecarlo_toys(
                num_toys, num_cores, num_reads=num_reads
            )
        return solution, error, cov, corr

    def compute_energy(self, x):
        x_binary = {}
        for bin in range(self.num_bins):
            num_bits = int(np.log2(self.upper_bounds[bin] + 1))
            bitstr = np.binary_repr(int(x[bin]), width=num_bits)
            for idx, bit in enumerate(bitstr[::-1]):
                x_binary[f"x{bin}[{idx}]"] = int(bit)
        energy = self.dwave_bqm.energy(sample=x_binary)
        return energy

    def solve_gurobi_integer(self):
        model = gurobipy.Model()
        model.setParam("OutputFlag", 0)
        x = [
            model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=ub)
            for ub in self.upper_bounds
        ]
        a = self.linear_coeffs
        B = self.quadratic_coeffs
        model.setObjective(a @ x + x @ B @ x, sense=gurobipy.GRB.MINIMIZE)
        model.optimize()
        solution = np.array([var.x for var in x])
        error = np.sqrt(solution)
        return solution, error

    def solve_gurobi_binary(self):
        model = gurobipy.Model()
        model.setParam("OutputFlag", 0)
        num_bits = [int(np.log2(ub + 1)) for ub in self.upper_bounds]
        x = [
            model.addVar(vtype=gurobipy.GRB.BINARY)
            for bin in range(self.num_bins)
            for _ in range(num_bits[bin])
        ]
        Q = self.qubo_matrix
        model.setObjective(x @ Q @ x, sense=gurobipy.GRB.MINIMIZE)
        model.optimize()
        bitstr = np.array([var.x for var in x], dtype=int)
        arrays = np.split(bitstr, np.cumsum(num_bits[:-1]))
        solution = np.array(
            [int("".join(arr.astype(str))[::-1], base=2) for arr in arrays], dtype=float
        )
        error = np.sqrt(solution)
        return solution, error
