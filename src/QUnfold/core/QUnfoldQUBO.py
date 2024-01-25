import numpy as np
from pyqubo import LogEncInteger
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import LeapHybridSampler
from dwave.system import DWaveSampler, EmbeddingComposite


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
        variables = [
            LogEncInteger(
                f"x{i}",
                value_range=(
                    -int(num_entries[i] * 0.2),
                    int(num_entries[i]) + int(num_entries[i] * 0.2) + 1,
                ),
            )
            for i in range(len(self.d))
        ]

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
            numpy.ndarray: the error of the unfolded distribution.
        """
        decoded_sampleset = self.model.decode_sampleset(sampleset)
        best_sample = min(decoded_sampleset, key=lambda s: s.energy)
        solution = np.array([best_sample.subh[label] for label in self.labels])
        error = np.sqrt(solution)

        return solution, error

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

    def solve_simulated_annealing(self, num_reads, seed=None):
        """
        Compute the unfolded distribution using simulated annealing.

        Args:
            num_reads (int): the number of reads used to approximate the solution.
            seed (int, optional): The seed used to randomize the reads. Defaults to None.

        Returns:
            numpy.ndarray: the unfolded distribution.
        """
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(self.bqm, num_reads=num_reads, seed=seed)
        solution, error = self._post_process_sampleset(sampleset=sampleset)
        return solution, error

    def solve_hybrid_sampler(self):
        """
        Compute the unfolded distribution using hybrid solver.

        Returns:
            numpy.ndarray: the unfolded distribution.
        """
        sampler = LeapHybridSampler()
        sampleset = sampler.sample(self.bqm)
        decoded_sample = self.model.decode_sampleset(sampleset)[0]
        solution = np.round(
            np.array([decoded_sample.subh[label] for label in self.labels])
        )
        error = np.sqrt(solution)
        return solution, error

    def solve_quantum_annealing(self, num_reads):
        """
        Compute the unfolded distribution using quantum annealing.

        Args:
            num_reads (int): the number of reads used to approximate the solution.

        Returns:
            numpy.ndarray: the unfolded distribution.
        """
        sampler = EmbeddingComposite(DWaveSampler())
        sampleset = sampler.sample(self.bqm, num_reads=num_reads)
        solution, error = self._post_process_sampleset(sampleset=sampleset)
        return solution, error

    def compute_energy(self, x):
        """
        Compute the energy of the Hamiltoninan for the given input histogram.

        Args:
            x (numpy.ndarray): input histogram.

        Returns:
            float: energy for the given input.
        """

        raise NotImplementedError("This feature is work in progress!")
