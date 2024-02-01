import numpy as np
from pyqubo import LogEncInteger
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import LeapHybridSampler
from dwave.system import DWaveSampler, EmbeddingComposite
import tqdm
from scipy.stats import chisquare


class QUnfoldQUBO:
    """
    Class used to perform the unfolding formulated as a QUBO problem.
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
        self.num_log_qubits = 0
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
        Build the Laplacian matrix.

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
        Get the vector of the number of the expected entries of the MC truth distribution.

        Returns:
            numpy.ndarray: the vector of the number of the expected entries of the MC truth distribution.
        """
        eff = np.sum(self.R, axis=0)
        eff[eff == 0] = 1  # Hack needed to not raise error
        num_entries = self.d / eff

        return num_entries

    def _define_variables(self):
        """
        Define the list of variables for the QUBO problem.

        Returns:
            list: encoded integer variables.
        """
        num_entries = self._get_expected_num_entries()
        variables = []
        for i in range(len(num_entries)):
            upper = int(num_entries[i] * 1.2) + 1  # Add 20% more
            var = LogEncInteger(label=f"x{i}", value_range=(0, upper))
            variables.append(var)
        return variables

    def _define_hamiltonian(self, x):
        """
        Define the Hamiltonian expression for the QUBO problem.

        Parameters:
            x (list): encoded integer variables.

        Returns:
            pyqubo.Expression: Hamiltonian expression.
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
        Process the sampleset and gives solution and error of the unfolding.

        Args:
            sampleset (sampleset): the sampleset to be processed.

        Returns:
            numpy.ndarray: the unfolded distribution.
        """
        best_sample = self.model.decode_sample(sampleset.first.sample, vartype="BINARY")
        solution = np.array([best_sample.subh[label] for label in self.labels])

        return solution

    def _compute_error(self, n_toys, solution, solver):
        """
        Compute the error of the unfolded distribution using pseudo-experiments in toy Monte Carlo simulations.

        Args:
            n_toys (int): the number of toy Monte Carlo experiments to be performed.
            solution (numpy.ndarray): the unfolded distribution.
            solver (func): a function that computes the unfolded distribution.

        Returns:
            numpy.ndarray: The errors associated with each bin in the unfolded distribution.

        Raises:
            ValueError: If the number of toys is not a positive integer.
        """

        if n_toys == 1:  # No toys case
            return np.sqrt(solution)
        else:  # Toys case
            unfolded_results = np.empty(shape=(n_toys, len(self.d)))
            for i in tqdm.trange(n_toys, desc="Running on toys"):
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
        Initialize QUBO model and BQM instance for the unfolding problem.
        """
        x = self._define_variables()
        h = self._define_hamiltonian(x)
        self.labels = [x[i].label for i in range(len(x))]
        self.model = h.compile()
        self.bqm = self.model.to_bqm()
        self.num_log_qubits = len(self.bqm.variables)

    def solve_simulated_annealing(self, num_reads, seed=None, n_toys=1):
        """
        Compute the unfolded distribution using simulated annealing.

        Args:
            num_reads (int): the number of reads used to approximate the solution.
            seed (int, optional): the seed used to randomize the reads. Defaults to None.
            n_toys (int, optional): the number of toys needed to compute the error of the unfolded distribution. If 1 the error is computed as the square-root of the number of entries in each bin, otherwise a statistical method based on toys is used. Defaults to 1.

        Returns:
            numpy.ndarray: the unfolded distribution.
        """

        def solver(unfolder):
            sampler = SimulatedAnnealingSampler()
            sampleset = sampler.sample(unfolder.bqm, num_reads=num_reads, seed=seed)
            solution = unfolder._post_process_sampleset(sampleset=sampleset)
            return solution

        self.solution = solver(self)
        error = self._compute_error(n_toys, self.solution, solver)

        return self.solution, error

    def solve_hybrid_sampler(self, n_toys=1):
        """
        Compute the unfolded distribution using hybrid solver.

        Args:
            n_toys (int, optional): the number of toys needed to compute the error of the unfolded distribution. If 1 the error is computed as the square-root of the number of entries in each bin, otherwise a statistical method based on toys is used. Defaults to 1.

        Returns:
            numpy.ndarray: the unfolded distribution.
        """

        def solver(unfolder):
            sampler = LeapHybridSampler()
            sampleset = sampler.sample(unfolder.bqm)
            decoded_sample = unfolder.model.decode_sampleset(sampleset)[0]
            solution = np.round(
                np.array([decoded_sample.subh[label] for label in unfolder.labels])
            )
            return solution

        self.solution = solver(self)
        error = self._compute_error(n_toys, self.solution, solver)

        return self.solution, error

    def solve_quantum_annealing(self, num_reads, n_toys=1):
        """
        Compute the unfolded distribution using quantum annealing.

        Args:
            num_reads (int): the number of reads used to approximate the solution.
            n_toys (int, optional): the number of toys needed to compute the error of the unfolded distribution. If 1 the error is computed as the square-root of the number of entries in each bin, otherwise a statistical method based on toys is used. Defaults to 1.

        Returns:
            numpy.ndarray: the unfolded distribution.
        """

        def solver(unfolder):
            sampler = EmbeddingComposite(DWaveSampler())
            sampleset = sampler.sample(unfolder.bqm, num_reads=num_reads)
            solution = unfolder._post_process_sampleset(sampleset=sampleset)
            return solution

        self.solution = solver(self)
        error = self._compute_error(n_toys, self.solution, solver)

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

        if method == "std":  # Use scipy
            chi2, _ = chisquare(
                self.solution,
                np.sum(self.solution) / np.sum(truth) * truth,
            )
        elif method == "cov":  # Use covariance matrix
            residuals = self.solution - truth
            inv_covariance_matrix = np.linalg.inv(self.cov_matrix)
            chi2 = residuals.T @ inv_covariance_matrix @ residuals

        dof = len(self.solution)
        return chi2 / dof

    def compute_energy(self, x):
        """
        Compute the energy of the Hamiltoninan for the given input histogram.

        Args:
            x (numpy.ndarray): input histogram.

        Returns:
            float: energy for the given input.
        """

        raise NotImplementedError("This feature is work in progress!")
