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
        TODO: docstring
        """
        efficiencies = np.sum(self.R, axis=0)
        num_entries = int(sum(self.d / efficiencies))
        return num_entries

    def _define_variables(self):
        """
        Define the list of variables for the QUBO problem.

        Returns:
            list: encoded integer variables.
        """
        num_entries = self._get_expected_num_entries()
        variables = [
            LogEncInteger(f"x{i}", value_range=(0, num_entries))
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
        TODO: docstring
        """
        median_energy = np.median(sampleset.record.energy)
        filtered_sampleset = sampleset.filter(pred=lambda s: s.energy < median_energy)
        results = np.array(
            [
                [sample.subh[label] for label in self.labels]
                for sample in self.model.decode_sampleset(filtered_sampleset)
            ]
        )
        solution = np.round(np.mean(results, axis=0))
        error = np.std(results, axis=0)
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

    def solve_simulated_annealing(self, num_reads, seed=None):
        """
        TODO: docstring
        """
        sampler = SimulatedAnnealingSampler()
        sampleset = sampler.sample(self.bqm, num_reads=num_reads, seed=seed)
        solution, error = self._post_process_sampleset(sampleset=sampleset)
        return solution, error

    def solve_hybrid_sampler(self):
        """
        TODO: docstring
        """
        sampler = LeapHybridSampler()
        sampleset = sampler.sample(self.bqm)
        decoded_sample = self.model.decode_sampleset(sampleset)[0]
        solution = np.round(
            np.array([decoded_sample.subh[label] for label in self.labels])
        )
        error = np.zeros(len(solution))
        return solution, error

    def solve_quantum_annealing(self, num_reads):
        """
        TODO: docstring
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
        num_entries = self._get_expected_num_entries()
        num_bits = int(np.ceil(np.log2(num_entries)))
        x_binary = {}
        for i, entry in enumerate(x):
            bitstr = np.binary_repr(int(entry), width=num_bits)
            for j, bit in enumerate(bitstr[::-1]):
                x_binary[f"x{i}[{j}]"] = int(bit)
        return self.bqm.energy(x_binary)
