import os
import sys
import numpy as np
import scipy as sp
import dimod
import minorminer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dwave.samplers import SimulatedAnnealingSampler
from dwave.samplers import SteepestDescentSolver
from dwave.system import LeapHybridSampler
from dwave.system import DWaveSampler, FixedEmbeddingComposite

try:
    import gurobipy
except ImportError:
    pass


class QUnfolder:
    def __init__(self, response, measured, binning, lam=0.0):
        self.R = response
        self.d = measured
        self.binning = binning
        self.lam = lam

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
        return [(int(np.ceil(np.log2(x * 1.2))) if x else 1) for x in exp]

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
        L = self._get_laplacian()
        return (self.R.T @ self.R) + self.lam * (L.T @ L)

    def _get_laplacian(self):
        n = self.num_bins
        L = np.zeros((n, n))
        x = self.binning[1:-2] + 0.5 * np.diff(self.binning[1:-1])
        k = 1
        for i in range(2, n - 2):
            xl, x0, xr = x[k - 1], x[k], x[k + 1]
            k += 1
            A = np.array([[xl**2, xl, 1], [x0**2, x0, 1], [xr**2, xr, 1]])
            coeffs_e0 = np.linalg.solve(A, [1, 0, 0])
            coeffs_e1 = np.linalg.solve(A, [0, 1, 0])
            coeffs_e2 = np.linalg.solve(A, [0, 0, 1])
            L[i, i - 1] += coeffs_e0[0]
            L[i - 1, i] += coeffs_e2[0]
            L[i, i] += 2 * coeffs_e1[0]
            L[i, i + 1] += coeffs_e2[0]
            L[i + 1, i] += coeffs_e0[0]
        L[1, 1] = L[2, 2]
        L[-2, -2] = L[-3, -3]
        L *= 2 / np.max(np.abs(L))
        return L

    def _get_qubo_matrix(self):
        dim = self.num_bins
        a = self.linear_coeffs
        B = self.quadratic_coeffs
        pvecs = self.precision_vectors
        linear_blocks = [a[i] * np.diag(pvecs[i]) for i in range(dim)]
        quadratic_blocks = [[B[i, j] * np.outer(pvecs[i], pvecs[j]) for j in range(dim)] for i in range(dim)]
        return sp.linalg.block_diag(*linear_blocks) + np.block(quadratic_blocks)

    def _get_dwave_bqm(self):
        Q = self.qubo_matrix
        size = len(Q)
        linear = {i: Q[i, i] for i in range(size)}
        quadratic = {(i, j): 2 * Q[i, j] for i in range(size) for j in range(i + 1, size)}
        return dimod.BinaryQuadraticModel(linear, quadratic, vartype=dimod.BINARY)

    def _decode_binary_solution(self, binsol):
        split_indices = np.cumsum(self.num_bits[:-1])
        bitstrings_list = np.split(binsol, indices_or_sections=split_indices)
        sol = np.array([pvec @ bits for pvec, bits in zip(self.precision_vectors, bitstrings_list)])
        return sol

    def _decode_binary_covariance(self, bincov):
        split_indices = np.cumsum(self.num_bits[:-1])
        rows = cols = np.split(np.arange(len(bincov)), indices_or_sections=split_indices)
        pvecs = self.precision_vectors
        nbins = self.num_bins
        cov = np.zeros(shape=(nbins, nbins))
        for i in range(nbins):
            for j in range(nbins):
                block = bincov[np.ix_(rows[i], cols[j])]
                cov[i, j] = pvecs[i] @ block @ pvecs[j]
        return cov

    def _post_process_sampleset(self, sampleset, steepest_descent):
        if steepest_descent:
            sampleset = SteepestDescentSolver().sample(self.dwave_bqm, initial_states=sampleset)
        solutions = np.array([rec.sample for rec in sampleset.record])
        energies = np.array([rec.energy for rec in sampleset.record])
        temperature = (np.max(energies) - np.min(energies)) / np.log(len(energies))
        weights = np.exp(-(1 / temperature) * (energies - np.min(energies)))
        binsol = np.average(solutions, weights=weights, axis=0)
        bincov = np.cov(solutions, rowvar=False, aweights=weights)
        sol = self._decode_binary_solution(binsol=binsol)
        cov = self._decode_binary_covariance(bincov=bincov)
        return sol, cov

    def _get_graph_embedding(self, **kwargs):
        source_edgelist = list(self.dwave_bqm.quadratic) + list((v, v) for v in self.dwave_bqm.linear)
        target_edgelist = self._sampler.edgelist
        return minorminer.find_embedding(S=source_edgelist, T=target_edgelist, **kwargs)

    def _run_montecarlo_toys(self, num_toys, prog_bar, num_cores, **kwargs):
        def run_toy(_):
            smeared_d = np.random.poisson(self.d)
            toy = QUnfolder(self.R, smeared_d, binning=self.binning, lam=self.lam)
            toy.initialize_qubo_model()
            if isinstance(self._sampler, DWaveSampler):
                embedding = toy._get_graph_embedding()
                sampler = FixedEmbeddingComposite(self._sampler, embedding=embedding)
            else:
                sampler = self._sampler
            sampleset = sampler.sample(toy.dwave_bqm, **kwargs)
            sol, _ = toy._post_process_sampleset(sampleset)
            return sol

        max_workers = num_cores if num_cores is not None else os.cpu_count()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            jobs = executor.map(run_toy, range(num_toys))
            desc = "Running MC toys"
            disable = not prog_bar
            results = list(tqdm(jobs, total=num_toys, desc=desc, disable=disable))
        cov_toys = np.cov(results, rowvar=False)
        return cov_toys

    def initialize_qubo_model(self):
        self.qubo_matrix = self._get_qubo_matrix()
        self.dwave_bqm = self._get_dwave_bqm()

    def solve_simulated_annealing(self, num_reads, num_toys=None, num_cores=None, seed=None):
        self._sampler = SimulatedAnnealingSampler()
        sampleset = self._sampler.sample(self.dwave_bqm, num_reads=num_reads, seed=seed)
        sol, cov = self._post_process_sampleset(sampleset, steepest_descent=True)
        if num_toys is not None:
            cov_toys = self._run_montecarlo_toys(num_toys, num_cores, num_reads=num_reads, seed=seed)
            cov += cov_toys
        return sol, cov

    def solve_hybrid_sampler(self, num_toys=None, num_cores=None):
        self._sampler = LeapHybridSampler()
        sampleset = self._sampler.sample(self.dwave_bqm)
        sol, cov = self._post_process_sampleset(sampleset, steepest_descent=False)
        if num_toys is not None:
            cov_toys = self._run_montecarlo_toys(num_toys, num_cores)
            cov += cov_toys
        return sol, cov

    def set_quantum_device(self, device_name=None, dwave_token=None):
        self._sampler = DWaveSampler(solver=device_name, token=dwave_token)

    def set_graph_embedding(self, **kwargs):
        self.graph_embedding = self._get_graph_embedding(**kwargs)

    def solve_quantum_annealing(self, num_reads, num_toys=None, prog_bar=True, num_cores=None):
        sampler = FixedEmbeddingComposite(self._sampler, embedding=self.graph_embedding)
        sampleset = sampler.sample(self.dwave_bqm, num_reads=num_reads)
        sol, cov = self._post_process_sampleset(sampleset, steepest_descent=True)
        if num_toys is not None:
            cov_toys = self._run_montecarlo_toys(num_toys, prog_bar, num_cores, num_reads=num_reads)
            cov += cov_toys
        return sol, cov

    def compute_energy(self, x):
        xbin = []
        num_bits = self.num_bits
        for i in range(self.num_bins):
            bitstr = bin(int(x[i]))[2:].zfill(num_bits[i])
            xbin.extend(map(int, bitstr[::-1]))
        energy = self.dwave_bqm.energy(sample=xbin)
        return energy

    if "gurobipy" in sys.modules:

        def solve_gurobi_integer(self):
            model = gurobipy.Model()
            model.setParam("OutputFlag", 0)
            x = [model.addVar(vtype=gurobipy.GRB.INTEGER, lb=0, ub=2**b - 1) for b in self.num_bits]
            a = self.linear_coeffs
            B = self.quadratic_coeffs
            model.setObjective(a @ x + x @ B @ x, sense=gurobipy.GRB.MINIMIZE)
            model.optimize()
            sol = np.array([var.x for var in x])
            cov = np.diag(sol)
            return sol, cov

        def solve_gurobi_binary(self):
            model = gurobipy.Model()
            model.setParam("OutputFlag", 0)
            num_bits = self.num_bits
            x = [model.addVar(vtype=gurobipy.GRB.BINARY) for i in range(self.num_bins) for _ in range(num_bits[i])]
            Q = self.qubo_matrix
            model.setObjective(x @ Q @ x, sense=gurobipy.GRB.MINIMIZE)
            model.optimize()
            bitstr = np.array([var.x for var in x], dtype=int)
            arrays = np.split(bitstr, np.cumsum(num_bits[:-1]))
            sol = np.array([int("".join(arr.astype(str))[::-1], base=2) for arr in arrays], dtype=float)
            cov = np.diag(sol)
            return sol, cov
