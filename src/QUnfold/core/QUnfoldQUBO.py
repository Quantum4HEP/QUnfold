import numpy as np
from tqdm import trange
from pyqubo import LogEncInteger
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import LeapHybridSampler
from dwave.system import DWaveSampler, EmbeddingComposite
from scipy.stats import chisquare


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
        self.cov_matrix = None
        self.corr_matrix = None
        self.solution = None

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

    def _define_hamiltonian(self, x):
        """
        Define the "integer" Hamiltonian expression of the QUBO problem.

        Parameters:
            x (list): "integer" QUBO problem variables.

        Returns:
            pyqubo.Expression: "integer" Hamiltonian expression.
        """
        hamiltonian = 0
        dim = len(x)
        a = -2 * (self.R.T @ self.d)
        for i in range(dim):
            hamiltonian += a[i] * x[i]
        G = self._get_laplacian(dim)
        B = (self.R.T @ self.R) + self.lam * (G.T @ G)
        for i in range(dim):
            for j in range(dim):
                hamiltonian += B[i, j] * x[i] * x[j]
        return hamiltonian

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

    def _compute_error(self, num_toys, solution, solver):
        """
        Compute the errors of the unfolded histogram by running Monte Carlo toy experiments.

        Args:
            num_toys (int): number of Monte Carlo toy experiments.
            solution (numpy.ndarray): unfolded histogram.
            solver (func): solver callable function.

        Returns:
            numpy.ndarray: errors on the unfolded histogram.
        """
        if num_toys == 1:
            return np.sqrt(solution)
        else:
            unfolded_results = np.empty(shape=(num_toys, len(self.d)))
            for i in trange(num_toys, desc="Running on toys"):
                smeared_d = np.random.poisson(self.d)
                unfolder = QUnfoldQUBO(self.R, smeared_d, lam=self.lam)
                unfolder.initialize_qubo_model()
                unfolded_results[i] = solver(unfolder)
            error = np.std(unfolded_results, axis=0)
            self.cov_matrix = np.cov(unfolded_results, rowvar=False)
            self.corr_matrix = np.corrcoef(unfolded_results, rowvar=False)
            return error

    def initialize_qubo_model(self):
        """
        Define the QUBO model and build the BQM instance for the unfolding problem.
        """
        x = self._define_variables()
        h = self._define_hamiltonian(x)
        self.labels = [x[i].label for i in range(len(x))]
        self.model = h.compile()
        self.bqm = self.model.to_bqm()
        self.num_log_qubits = len(self.bqm.variables)

    def solve_simulated_annealing(self, num_reads, num_toys=1, seed=None):
        """
        Compute the unfolded histogram by running DWave simulated annealing sampler.

        Args:
            num_reads (int): number of sampler runs per toy experiment.
            num_toys (int, optional): number of Monte Carlo toy experiments (default is 1).
            seed (int, optional): random seed (defaults is None).

        Returns:
            numpy.ndarray: unfolded histogram.
        """

        def solver(unfolder):
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(unfolder.bqm, num_reads=num_reads, seed=seed)
            return unfolder._post_process_sampleset(sampleset)

        self.solution = solver(unfolder=self)
        error = self._compute_error(num_toys, self.solution, solver)
        return self.solution, error

    def solve_hybrid_sampler(self, num_toys=1):
        """
        Compute the unfolded histogram by running DWave hybrid sampler.

        Args:
            num_toys (int, optional): number of Monte Carlo toy experiments (default is 1).

        Returns:
            numpy.ndarray: unfolded histogram.
        """

        def solver(unfolder):
            sampler = LeapHybridSampler()
            sampleset = sampler.sample(unfolder.bqm)
            return unfolder._post_process_sampleset(sampleset)

        self.solution = solver(unfolder=self)
        error = self._compute_error(num_toys, self.solution, solver)
        return self.solution, error

    def solve_quantum_annealing(self, num_reads, num_toys=1):
        """
        Compute the unfolded histogram by running DWave quantum annealing sampler.

        Args:
            num_reads (int): number of sampler runs per toy experiment.
            num_toys (int, optional): number of Monte Carlo toy experiments (default is 1).

        Returns:
            numpy.ndarray: unfolded histogram.
        """

        def solver(unfolder):
            sampler = EmbeddingComposite(DWaveSampler())
            sampleset = sampler.sample(unfolder.bqm, num_reads=num_reads)
            return unfolder._post_process_sampleset(sampleset)

        self.solution = solver(unfolder=self)
        error = self._compute_error(num_toys, self.solution, solver)
        return self.solution, error

    def compute_chi2(self, truth, method="std"):
        """
        Compute the chi-square statistic for the unfolded distribution.

        Args:
            truth (array-like): The true distribution against which to compare.
            method (str, optional): Method for computing chi-square. Options: "std" (use scipy) or "cov" (use covariance matrix). Default is "std".

        Returns:
            float or None: The computed chi-square statistic, or None if the method is not recognized.
        """
        chi2 = None
        null_indices = truth == 0
        truth[null_indices] += 1
        self.solution[null_indices] += 1
        if method == "std":
            chi2, _ = chisquare(
                self.solution,
                np.sum(self.solution) / np.sum(truth) * truth,
            )
        elif method == "cov":
            residuals = self.solution - truth
            inv_covariance_matrix = np.linalg.inv(self.cov_matrix)
            chi2 = residuals.T @ inv_covariance_matrix @ residuals
        dof = len(self.solution)
        return chi2 / dof

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
