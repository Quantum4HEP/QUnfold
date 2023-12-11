import numpy as np
from pyqubo import LogEncInteger
from dwave.samplers import SimulatedAnnealingSampler
from dwave.system import LeapHybridSampler


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

    def _define_variables(self, opt):
        """
        Define the list of variables for the QUBO problem.

        Parameters:
            opt (bool): enables optimal range for the integer variables.

        Returns:
            list: encoded integer variables.
        """
        variables = []
        if opt:
            for i, mean in enumerate(self.d):
                error = 5 * mean**0.5
                lower = 0 if mean - error < 0 else int(np.floor(mean - error))
                upper = 1 if mean == 0 else int(np.ceil(mean + error))
                variables.append(LogEncInteger(f"x{i}", value_range=(lower, upper)))
        else:
            variables = [
                LogEncInteger(f"x{i}", value_range=(0, int(sum(self.d))))
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

        # Add linear terms
        a = -2 * (self.R.T @ self.d)
        for i in range(dim):
            hamiltonian += a[i] * x[i]

        # Add quadratic terms:
        G = self._get_laplacian(dim)
        B = (self.R.T @ self.R) + self.lam * (G.T @ G)
        for i in range(dim):
            for j in range(dim):
                hamiltonian += B[i, j] * x[i] * x[j]
        return hamiltonian

    def initialize_qubo_model(self, optimize_vars_range=False):
        """
        Initialize QUBO model and BQM instance for the unfolding problem.

        Parameters:
            optimize_vars_range (bool, optional): enables optimal range for the integer variables (default is False).
        """
        x = self._define_variables(opt=optimize_vars_range)
        h = self._define_hamiltonian(x)
        self.labels = [x[i].label for i in range(len(x))]
        self.model = h.compile()
        self.bqm = self.model.to_bqm()

    def solve_simulated_annealing(self, num_reads=1, seed=None):
        """
        TODO: docstring
        """
        sampler = SimulatedAnnealingSampler()
        sample_set = sampler.sample(self.bqm, num_reads=num_reads, seed=seed)
        solution_set = np.array(
            [
                [sample.subh[label] for label in self.labels]
                for sample in self.model.decode_sampleset(sample_set)
            ]
        )
        solution = np.round(np.mean(solution_set, axis=0))
        cov_matrix = np.cov(solution_set, rowvar=False, ddof=1)
        error = np.sqrt(np.diagonal(cov_matrix) / num_reads)
        return solution, error

    def solve_quantum_annealing(self, num_reads):
        """
        TODO: docstring
        """
        pass

    def solve_hybrid_sampler(self):
        """
        Solve the unfolding QUBO problem using the Leap Hybrid sampler.

        Returns:
            numpy.ndarray: unfolded histogram.
        """
        # TODO: fix solution selection and error computation
        sampler = LeapHybridSampler()
        sampleset = sampler.sample(self.bqm)
        decoded_sampleset = self.model.decode_sampleset(sampleset)
        best_sample = min(decoded_sampleset, key=lambda s: s.energy)
        return np.array([best_sample.subh[label] for label in self.labels])

    def compute_energy(self, x):
        """
        Compute the energy of the Hamiltoninan for the given input histogram.

        Args:
            x (numpy.ndarray): input histogram.

        Returns:
            float: energy for the given input.
        """
        # TODO: fix for the case 'optimize_vars_range=True'
        num_bits = int(np.floor(np.log2(sum(self.d))))
        x_binary = {}
        for i, entry in enumerate(x):
            bitstr = np.binary_repr(int(entry), width=num_bits)
            for j, bit in enumerate(bitstr[::-1]):
                x_binary[f"x{i}[{j}]"] = int(bit)
        return self.bqm.energy(x_binary)
