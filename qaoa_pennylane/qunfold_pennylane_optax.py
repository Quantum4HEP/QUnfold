import pennylane as qml
import pennylane.numpy as np
import jax
import jax.numpy as jnp
import optax

a_vector = np.loadtxt("a_vector.txt")
B_matrix = np.loadtxt("B_matrix.txt")
upper_bounds = np.loadtxt("upper_bounds.txt")
print("\nUpper bounds =")
print(upper_bounds)

qubits_per_var = [int(np.log2(ub + 1)) for ub in upper_bounds]
print("\nQubits per var =")
print(qubits_per_var)

num_wires = sum(qubits_per_var)
dev = qml.device("default.qubit", wires=num_wires)
p = depth = 1

def qubo_hamitonian(a, B):
    coeffs_linear = []
    obs_linear = []
    for i in range(num_wires):
        if a[i] != 0:
            coeffs_linear.append(a[i])
            obs_linear.append(qml.PauliZ(i))
    H_linear = qml.Hamiltonian(coeffs=coeffs_linear, observables=obs_linear)
    coeffs_quadratic = []
    obs_quadratic = []
    for i in range(num_wires-1):
        for j in range(i+1, num_wires):
            if B[i, j] != 0:
                coeffs_quadratic.append(B[i, j])
                obs_quadratic.append(qml.PauliZ(i) @ qml.PauliZ(j))
    H_quadratic = qml.Hamiltonian(coeffs=coeffs_quadratic, observables=obs_quadratic)
    H_qubo = H_linear + H_quadratic
    return H_qubo

H_qubo = qubo_hamitonian(a=a_vector, B=B_matrix)

def mixer_hamiltonian():
    coeffs = jnp.ones(len(dev.wires))
    obs = [qml.PauliX(i) for i in dev.wires]
    H_mixer = qml.Hamiltonian(coeffs=coeffs, observables=obs)
    return H_mixer

H_mixer = mixer_hamiltonian()

def qaoa_layer(gamma, alpha):
    qml.qaoa.cost_layer(gamma=gamma, hamiltonian=H_qubo)
    qml.qaoa.mixer_layer(alpha=alpha, hamiltonian=H_mixer)

def var_circuit(params):
    for wire in dev.wires:
        qml.Hadamard(wires=wire)
    gamma, alpha = params
    qml.layer(qaoa_layer, depth, gamma, alpha)

@jax.jit
@qml.qnode(device=dev)
def cost_function(params):
    var_circuit(params=params)
    return qml.expval(H_qubo)

opt = optax.adam(learning_rate=0.3)

@jax.jit
def update_step(_, args):
    params, opt_state = args
    _, grad = jax.value_and_grad(cost_function)(params)
    updates, opt_state = opt.update(grad, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state

@jax.jit
def run_optimization(params, iterations):
    opt_state = opt.init(params)
    args = params, opt_state
    params, opt_state = jax.lax.fori_loop(0, iterations, update_step, args)
    return params

init_params = jnp.array([[0.5]*depth, [0.5]*depth])
print("\nCompiling the optimization routine...")
run_optimization(params=init_params, iterations=1)
print("Done")

iterations = 100
fin_params = run_optimization(params=init_params, iterations=iterations)
print("\nFinal params =")
print(fin_params)

dev = qml.device("default.qubit", wires=num_wires, shots=1)
@qml.qnode(device=dev)
def sample_circuit(params):
    var_circuit(params)
    return qml.sample()

sample = sample_circuit(params=fin_params)
bitlist = [str(bit) for bit in sample]
histogram = []
start = 0
end = 0
for shift in qubits_per_var:
    end += shift
    bitstring = "".join(bitlist[start:end])
    histogram.append(int(bitstring, base=2))
    start += shift
print("\nOutput histogram =")
print(histogram)
