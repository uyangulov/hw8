from tensor_contraction import TensorNetwork, TensorNetworkOptimizer, State
import numpy as np
import time
from itertools import product
import csv


def generate_lambda(probability, n):
    lambd = np.zeros((n, n))
    for i in range(n):
        for j in range(i):
            if np.random.rand() < probability:
                lambd[i, j] = 1
    return lambd

def run_optimization(lambd, n_layers, k=1):
    net = TensorNetwork.qaoa_like_tree_only(lambd=lambd, n_layers=n_layers, generate_meas_layer=True)
    opt = TensorNetworkOptimizer(net)
    start_time = time.perf_counter()
    seq, _ = opt.optimize_network(k)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, seq

def run_contaction(lambd, gammas, betas, seq):
    net = TensorNetwork.qaoa_like_from_params(lambd=lambd, gammas=gammas, betas=betas, generate_meas_layer=True)
    start_time = time.perf_counter()
    for (i, j) in seq:
        net.merge_nodes(i, j, actually_contract=True)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, net.nodes[0].tensor

def run_circuit(lambd, gammas, betas):
    state = State(lambd.shape[0])
    start_time = time.perf_counter()
    state.run_qaoa(lambd, gammas, betas)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, state.state


n_layers = 1
n_trials = 10
N_values = np.arange(1, 22, 3)
print(N_values)
probabilities = [0.1, 0.3, 0.5, .7, 1]

gammas = np.random.uniform(0, 2*np.pi, size=n_layers)
betas = np.random.uniform(0, 2*np.pi, size=n_layers)


max_attempts = 10
csv_filename = "optimization_and_contraction_times.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "N", "probability",
        "opt_time_mean", "opt_time_std",
        "conv_time_mean", "conv_time_std",
        "circuit_time_mean", "circuit_time_std"
    ])

    seen_lambdas = set()

    for N, probability in product(N_values, probabilities):
        opt_times = []
        conv_times = []
        circuit_times = []
        seen_lambdas.clear()
        attempts = 0
        
        # no need to sample if prob = 1
        n_trial_this = n_trials if probability < 1.0 else 1
        while len(opt_times) < n_trial_this and attempts < max_attempts:
            
            # lambdas are cached to avoid repeating qaoa problems
            lambd = generate_lambda(probability, N)
            if lambd.tobytes() in seen_lambdas:
                attempts += 1
                continue

            seen_lambdas.add(lambd.tobytes())
            opt_time, seq = run_optimization(lambd, n_layers)
            conv_time, out_tensor = run_contaction(lambd, gammas, betas, seq)
            circuit_time, out_state = run_circuit(lambd, gammas, betas)

            opt_times.append(opt_time)
            conv_times.append(conv_time)
            circuit_times.append(circuit_time)

        opt_mean = np.mean(opt_times)
        opt_std = np.std(opt_times)
        conv_mean = np.mean(conv_times)
        conv_std = np.std(conv_times)
        circuit_mean = np.mean(circuit_time)
        circuit_std = np.std(circuit_times)

        print(
            f"N={N}, prob={probability} | opt_mean={opt_mean:.4f}s, conv_mean={conv_mean:.4f}s", f"circuit_mean = {circuit_mean:.4f}s")
        writer.writerow([N, probability, opt_mean, opt_std, conv_mean, conv_std, circuit_mean, circuit_std])