import os
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from pyqubo import LogEncInteger
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import LeapHybridSampler
from dwave.system import DWaveSampler, EmbeddingComposite


class QUnfoldQUBO:
    """
    Class to solve the unfolding problem formulated as a QUBO model.
    """

    def __init__(self, response=None, measured=None, lam=0.0):
        """
        Initialize the QUnfoldQUBO object.

        Parameters:
            response (numpy.ndarray, optional): input response matrix (default is None).
            measured (numpy.ndarray, optional): input measured histogram (default is None).
            lam (float, optional): regularization parameter (default is 0.0).
        """
        self.R = response
        self.d = measured
        self.lam = lam

    def set_response(self, response):
        """
        Set the input response matrix.

        Parameters:
            response (numpy.ndarray): input response matrix.
        """
        self.R = response

    def set_measured(self, measured):
        """
        Set the input measured histogram.

        Parameters:
            measured (numpy.ndarray): input measured histogram.
        """
        self.d = measured

    def set_lam_parameter(self, lam):
        """
        Set the lambda regularization parameter.

        Parameters:
            lam (float): regularization parameter.
        """
        self.lam = lam

    @staticmethod
    def _get_laplacian(dim):
        """
        Get the Laplacian matrix (discrete 2nd-order derivative).

        Parameters:
            dim (int): dimension of the matrix.

        Returns:
            numpy.ndarray: Laplacian matrix.
        """
        diag = np.array([-1] + [-2] * (dim - 2) + [-1])
        D = np.diag(diag).astype(float)
        diag1 = np.ones(dim - 1)
        D += np.diag(diag1, k=1) + np.diag(diag1, k=-1)
        return D

    def _get_expected_num_entries(self):
        """
        Get the array of the expected number of entries in each bin.

        Returns:
            numpy.ndarray: expected histogram.
        """
        effs = np.sum(self.R, axis=0)
        effs[effs == 0] = 1
        num_entries = self.d / effs
        return num_entries

    def _define_variables(self):
        """
        Define the binary encoded "integer" variables of the QUBO problem.

        Returns:
            list: "integer" QUBO problem variables.
        """
        num_entries = self._get_expected_num_entries()
        variables = []
        for i in range(len(num_entries)):
            upper = int(2 ** np.ceil(np.log2((num_entries[i] + 1) * 1.2))) - 1
            var = LogEncInteger(label=f"x{i}", value_range=(0, upper))
            variables.append(var)
        return variables

    def _get_linear_array(self):
        """
        Compute the linear terms coefficients of the "integer" Hamiltonian expression.

        Returns:
            numpy.ndarray: linear terms coefficients.
        """
        return -2 * (self.R.T @ self.d)

    def _get_quadratic_matrix(self):
        """
        Compute the quadratic terms coefficients of the "integer" Hamiltonian expression.

        Returns:
            numpy.ndarray: quadratic terms coefficients.
        """
        G = self._get_laplacian(dim=len(self.d))
        return (self.R.T @ self.R) + self.lam * (G.T @ G)

    def _define_hamiltonian(self, x):
        """
        Define the "integer" Hamiltonian expression of the QUBO problem.

        Parameters:
            x (list): "integer" QUBO problem variables.

        Returns:
            pyqubo.Expression: "integer" Hamiltonian expression.
        """
        hamiltonian = 0
        num_bins = len(self.d)
        a = self._get_linear_array()
        for i in range(num_bins):
            hamiltonian += a[i] * x[i]
        B = self._get_quadratic_matrix()
        for i in range(num_bins):
            for j in range(num_bins):
                hamiltonian += B[i, j] * x[i] * x[j]
        return hamiltonian

    def _get_qubo_matrix(self):
        """
        Get the QUBO matrix from the BQM instance of the unfolding problem.

        Returns:
            numpy.ndarray: QUBO problem matrix.
        """
        qubo_dict, _ = self.bqm.to_qubo()
        variables = sorted(self.bqm.variables)
        num_vars = len(variables)
        Q = np.zeros(shape=(num_vars, num_vars))
        for i in range(num_vars):
            for j in range(i, num_vars):
                key = (variables[i], variables[j])
                if key in qubo_dict:
                    Q[i, j] = Q[j, i] = qubo_dict[key]
                else:
                    Q[i, j] = Q[j, i] = qubo_dict[key[::-1]]
        return Q

    def _post_process_sampleset(self, sampleset):
        """
        Post-process the output sampleset selecting and decoding the lower-energy sample.

        Args:
            sampleset (sampleset): set of output samples.

        Returns:
            numpy.ndarray: unfolded histogram.
        """
        sample = self.model.decode_sample(sampleset.first.sample, vartype="BINARY")
        solution = np.array([sample.subh[label] for label in self.labels])
        return solution

    def _compute_stat_errors(self, solver, solution, num_toys, num_cores):
        """
        Compute the statistical errors on the unfolded histogram by running Monte Carlo toy experiments.

        Args:
            solver (func): function to compute the unfolded histogram.
            solution (numpy.ndarray): unfolded histogram.
            num_toys (int): number of Monte Carlo toy experiments.
            num_cores (int): number of CPU cores to use for running Monte Carlo toy experiments in parallel.

        Returns:
            numpy.ndarray: errors on the unfolded histogram.
            numpy.ndarray: covariance matrix.
            numpy.ndarray: correlation matrix.
        """

        def toy_job(i):
            smeared_d = np.random.poisson(self.d)
            unfolder = QUnfoldQUBO(self.R, smeared_d, lam=self.lam)
            unfolder.initialize_qubo_model()
            unfolded_results[i] = solver(unfolder)

        num_cores = num_cores if num_cores is not None else os.cpu_count() - 2
        if num_toys == 1:
            return np.sqrt(solution), None, None
        else:
            unfolded_results = np.empty(shape=(num_toys, len(self.d)))
            with ThreadPoolExecutor(max_workers=num_cores) as executor:
                list(
                    tqdm(
                        executor.map(toy_job, range(num_toys)),
                        total=num_toys,
                        desc="Running on toys",
                    )
                )
            cov_matrix = np.cov(unfolded_results, rowvar=False)
            errors = np.sqrt(np.diag(cov_matrix))
            corr_matrix = np.corrcoef(unfolded_results, rowvar=False)
            return errors, cov_matrix, corr_matrix

    def initialize_qubo_model(self):
        """
        Define the QUBO model and build the BQM instance for the unfolding problem.
        """
        x = self._define_variables()
        h = self._define_hamiltonian(x)
        self.labels = [x[i].label for i in range(len(x))]
        self.model = h.compile()
        self.bqm = self.model.to_bqm()
        self.qubo_matrix = self._get_qubo_matrix()

    def solve_simulated_annealing(
        self, num_reads, num_toys=1, num_cores=None, seed=None
    ):
        """
        Compute the unfolded histogram by running DWave simulated annealing sampler.

        Args:
            num_reads (int): number of sampler runs per Monte Carlo toy experiment.
            num_toys (int, optional): number of Monte Carlo toy experiments (default is 1).
            num_cores (int, optional): number of CPU cores used to compute the toys in parallel (default is None).
            seed (int, optional): random seed (defaults is None).

        Returns:
            numpy.ndarray: unfolded histogram.
            numpy.ndarray: errors on the unfolded histogram.
            numpy.ndarray: covariance matrix.
            numpy.ndarray: correlation matrix.
        """

        def solver(unfolder):
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(unfolder.bqm, num_reads=num_reads, seed=seed)
            return unfolder._post_process_sampleset(sampleset)

        solution = solver(unfolder=self)
        errors, cov_matrix, corr_matrix = self._compute_stat_errors(
            solver, solution, num_toys, num_cores
        )
        return solution, errors, cov_matrix, corr_matrix

    def solve_hybrid_sampler(self, num_toys=1, num_cores=None):
        """
        Compute the unfolded histogram by running DWave hybrid sampler.

        Args:
            num_toys (int, optional): number of Monte Carlo toy experiments (default is 1).
            num_cores (int, optional): number of CPU cores used to compute the toys in parallel (default is None).

        Returns:
            numpy.ndarray: unfolded histogram.
            numpy.ndarray: errors on the unfolded histogram.
            numpy.ndarray: covariance matrix.
            numpy.ndarray: correlation matrix.
        """

        def solver(unfolder):
            sampler = LeapHybridSampler()
            sampleset = sampler.sample(unfolder.bqm)
            return unfolder._post_process_sampleset(sampleset)

        solution = solver(unfolder=self)
        errors, cov_matrix, corr_matrix = self._compute_stat_errors(
            solver, solution, num_toys, num_cores
        )
        return solution, errors, cov_matrix, corr_matrix

    def solve_quantum_annealing(self, num_reads, num_toys=1, num_cores=None):
        """
        Compute the unfolded histogram by running DWave quantum annealing sampler.

        Args:
            num_reads (int): number of sampler runs per toy experiment.
            num_toys (int, optional): number of Monte Carlo toy experiments (default is 1).
            num_cores (int, optional): number of CPU cores used to compute the toys in parallel (default is None).

        Returns:
            numpy.ndarray: unfolded histogram.
            numpy.ndarray: errors on the unfolded histogram.
            numpy.ndarray: covariance matrix.
            numpy.ndarray: correlation matrix.
        """

        def solver(unfolder):
            sampler = EmbeddingComposite(DWaveSampler())
            sampleset = sampler.sample(unfolder.bqm, num_reads=num_reads)
            return unfolder._post_process_sampleset(sampleset)

        solution = solver(unfolder=self)
        errors, cov_matrix, corr_matrix = self._compute_stat_errors(
            solver, solution, num_toys, num_cores
        )
        return solution, errors, cov_matrix, corr_matrix

    def compute_energy(self, x):
        """
        Compute the energy of the QUBO Hamiltoninan for the given input histogram.

        Args:
            x (numpy.ndarray): input histogram.

        Returns:
            float: QUBO Hamiltonian energy.
        """
        num_entries = self._get_expected_num_entries()
        x_binary = {}
        for i in range(len(x)):
            num_bits = int(np.ceil(np.log2((num_entries[i] + 1) * 1.2)))
            bitstr = np.binary_repr(int(x[i]), width=num_bits)
            for j, bit in enumerate(bitstr[::-1]):
                x_binary[f"x{i}[{j}]"] = int(bit)
        return self.bqm.energy(x_binary)
