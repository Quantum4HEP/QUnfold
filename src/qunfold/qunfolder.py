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
from dwave.system import LeapHybridCQMSampler
from dwave.system import DWaveSampler, FixedEmbeddingComposite
from dimod import Integer, ConstrainedQuadraticModel

import inspect

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
        self.sol_pick = "lowest-energy"


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
        x = (self.binning[:-1] + self.binning[1:]) / 2
        n = len(x)
        L = np.zeros(shape=(n, n))
        for i in range(1, n - 1):
            h1 = x[i] - x[i - 1]
            h2 = x[i + 1] - x[i]
            L[i, i - 1] = 2 / (h1 * (h1 + h2))
            L[i, i] = -2 / (h1 * h2)
            L[i, i + 1] = 2 / (h2 * (h1 + h2))
        L[0, :] = L[1, :]
        L[-1, :] = L[-2, :]
        L = 2 * L / np.max(np.abs(L))
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

    def _decode_array(self, arr):
        split_indices = np.cumsum(self.num_bits[:-1])
        bitstrings_list = np.split(arr, indices_or_sections=split_indices)
        decoded_arr = np.array([pvec @ bits for pvec, bits in zip(self.precision_vectors, bitstrings_list)])
        return decoded_arr

    def _decode_matrix(self, mat):
        split_indices = np.cumsum(self.num_bits[:-1])
        rows = cols = np.split(np.arange(len(mat)), indices_or_sections=split_indices)
        pvecs = self.precision_vectors
        nbins = self.num_bins
        decoded_mat = np.zeros(shape=(nbins, nbins))
        for i in range(nbins):
            for j in range(nbins):
                block = mat[np.ix_(rows[i], cols[j])]
                decoded_mat[i, j] = pvecs[i] @ block @ pvecs[j]
        return decoded_mat

    def _post_process_sampleset(self, sampleset):
        sampleset = SteepestDescentSolver().sample(self.dwave_bqm, initial_states=sampleset)
        solutions = np.array([rec.sample for rec in sampleset.record])
        energies = np.array([rec.energy for rec in sampleset.record])
        if self.sol_pick == "mean":
            binsol = np.mean(solutions, axis=0)
            devs = solutions - binsol
            bincov = (devs.T @ devs) / len(devs)
            sol = self._decode_array(arr=binsol)
            cov = self._decode_matrix(mat=bincov)
        elif self.sol_pick == "weighted-average":
            temperature = (np.max(energies) - np.min(energies)) / np.log(len(energies))
            weights = np.exp(-(1 / temperature) * (energies - np.min(energies)))
            weights /= np.sum(weights)
            binsol = np.average(solutions, weights=weights, axis=0)
            devs = solutions - binsol
            bincov = (devs.T @ (devs * weights[:, np.newaxis])) / (1 - np.sum(weights**2))
            sol = self._decode_array(arr=binsol)
            cov = self._decode_matrix(mat=bincov)
        elif self.sol_pick == "lowest-energy":
            binsol = solutions[np.argmin(energies)]
            sol = self._decode_array(arr=binsol)
            cov = np.diag(sol)
        sol = np.round(sol)
        return sol, cov

    def _get_graph_embedding(self, **kwargs):
        source_edgelist = list(self.dwave_bqm.quadratic) + list((v, v) for v in self.dwave_bqm.linear)
        target_edgelist = self._sampler.edgelist
        return minorminer.find_embedding(S=source_edgelist, T=target_edgelist, **kwargs)

    def _run_montecarlo_toys(self, num_toys, num_cores,prog_bar=True, **kwargs):
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

        def run_toy_multithread(max_workers=1,num_toys=num_toys):
            max_workers = num_cores if num_cores is not None else os.cpu_count()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                jobs = executor.map(run_toy, range(num_toys))
                desc = "Running MC toys"
                disable = not prog_bar
                results = list(tqdm(jobs, total=num_toys, desc=desc, disable=disable))
            cov_toys = np.cov(results, rowvar=False)
            return cov_toys
        return run_toy_multithread()

    def initialize_qubo_model(self):
        self.qubo_matrix = self._get_qubo_matrix()
        self.dwave_bqm = self._get_dwave_bqm()

    def solve_simulated_annealing(self, num_reads, num_toys=None, num_cores=None, seed=None):
        self._sampler = SimulatedAnnealingSampler()
        sampleset = self._sampler.sample(self.dwave_bqm, num_reads=num_reads, seed=seed)
        sol, cov = self._post_process_sampleset(sampleset)
        if num_toys is not None:
            cov_toys = self._run_montecarlo_toys(num_toys, num_cores, num_reads=num_reads, seed=seed)
            np.add(cov, cov_toys, out=cov, casting="unsafe")
            #cov += cov_toys
        return sol, cov

    def simsamplwrapper(self,*args,**kwargs):
        return self._sampler.sample(bqm=kwargs['bqm'],num_reads=kwargs['num_reads'],seed=kwargs['seed'])

    def distributed_simulated_annealing(self, num_reads, client, num_toys=None, num_cores=None, seed=None):
        self._sampler = SimulatedAnnealingSampler()
        
        fixed_args = {"bqm" : self.dwave_bqm, "num_reads":1,"seed":seed}
        num_calls = range(int(num_reads/100))
        handle = client.map(self.simsamplwrapper, num_calls, bqm=self.dwave_bqm, num_reads=100, seed=seed )

        sampleset = client.gather(handle)
        combined_sampleset = dimod.concatenate(sampleset)
        
        sol, cov = self._post_process_sampleset(combined_sampleset)
        
        #if num_toys is not None:
        #    cov_toys = self._run_montecarlo_toys(num_toys, num_cores, num_reads=num_reads, seed=seed)
        #    np.add(cov, cov_toys, out=cov, casting="unsafe")
        #    #cov += cov_toys

        return sol, cov

    def solve_hybrid_sampler(self):
        qm = ConstrainedQuadraticModel()
        x = np.array([Integer(f"x{i}", upper_bound=2**nb - 1) for i, nb in enumerate(self.num_bits)])
        objective = (self.R @ x - self.d) @ (self.R @ x - self.d)
        if self.lam != 0:
            G = self._get_laplacian()
            objective += self.lam * (G @ x) @ (G @ x)
        qm.set_objective(objective)
        sampler = LeapHybridCQMSampler()
        sampleset = sampler.sample_cqm(qm)
        sol = np.array([sampleset.first.sample[var] for var in qm.variables])
        cov = np.diag(sol)
        return sol, cov

    def set_quantum_device(self, device_name=None):
        self._sampler = DWaveSampler(solver=device_name)

    def set_graph_embedding(self, graph_embedding=None, **kwargs):
        self.graph_embedding = graph_embedding if graph_embedding is not None else self._get_graph_embedding(**kwargs)

    def solve_quantum_annealing(self, num_reads, num_toys=None, prog_bar=True, num_cores=None):
        sampler = FixedEmbeddingComposite(self._sampler, embedding=self.graph_embedding)
        sampleset = sampler.sample(self.dwave_bqm, num_reads=num_reads)
        sol, cov = self._post_process_sampleset(sampleset)
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
            vtype = gurobipy.GRB.INTEGER
            sense = gurobipy.GRB.MINIMIZE
            model.setParam("OutputFlag", 0)
            x = [model.addVar(vtype=vtype, lb=0, ub=2**b - 1) for b in self.num_bits]
            R, d = self.R, self.d
            objective = (R @ x - d) @ (R @ x - d)
            if self.lam != 0:
                G = self._get_laplacian()
                objective += self.lam * (G @ x) @ (G @ x)
            model.setObjective(objective, sense=sense)
            model.optimize()
            sol = np.array([var.x for var in x])
            cov = np.diag(sol)
            return sol, cov

        def solve_gurobi_binary(self):
            model = gurobipy.Model()
            vtype = gurobipy.GRB.BINARY
            sense = gurobipy.GRB.MINIMIZE
            model.setParam("OutputFlag", 0)
            x = [model.addVar(vtype=vtype) for i in range(self.num_bins) for _ in range(self.num_bits[i])]
            Q = self.qubo_matrix
            model.setObjective(x @ Q @ x, sense=sense)
            model.optimize()
            bitstr = np.array([var.x for var in x], dtype=int)
            arrays = np.split(bitstr, np.cumsum(self.num_bits[:-1]))
            sol = np.array([int("".join(arr.astype(str))[::-1], base=2) for arr in arrays], dtype=float)
            cov = np.diag(sol)
            return sol, cov
