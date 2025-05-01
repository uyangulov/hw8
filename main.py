from tensor_contraction import TensorNetwork
from greedy_optimizer import TensorNetworkOptimizer
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


def measure_optimization_time(lambd, n_layers, k=1):
    net = TensorNetwork.qaoa_like_tree_only(lambd=lambd, n_layers=n_layers)
    opt = TensorNetworkOptimizer(net)
    start_time = time.perf_counter()
    seq, cost = opt.optimize_network(k)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time, seq, cost


def measure_conv_time(lambd, gammas, betas, n_layers, seq, k=1):
    gammas = np.random.uniform(0, np.pi, size=n_layers)
    betas = np.random.uniform(0, np.pi, size=n_layers)
    net = TensorNetwork.qaoa_like_from_params(
        lambd=lambd, gammas=gammas, betas=betas)
    start_time = time.perf_counter()
    for (i, j) in seq:
        net.merge_nodes(i, j)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time

def measure_circuit_time(lambd, n_layers):


n_layers = 3
n_trials = 5
N_values = np.arange(2, 11)
probabilities = [0.3, 0.5, 0.7, 1.0]


max_attempts = 50
csv_filename = "optimization_and_contraction_times.csv"

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "N", "probability",
        "opt_time_mean", "opt_time_std",
        "conv_time_mean", "conv_time_std",
        "status"
    ])

    seen_lambdas = set()

    for N, probability in product(N_values, probabilities):
        opt_times = []
        conv_times = []
        seen_lambdas.clear()
        attempts = 0

        # no need to sample if prob = 1
        n_trial_this = n_trials if probability < 1.0 else 1
        while len(opt_times) < n_trial_this and attempts < max_attempts:

            lambd = generate_lambda(probability, N)
            lambd_hash = hash(lambd.tobytes())
            if lambd_hash in seen_lambdas:
                attempts += 1
                continue

            seen_lambdas.add(lambd_hash)
            opt_time, seq, cost = measure_optimization_time(lambd, n_layers)
            conv_time = measure_conv_time(lambd, n_layers, seq)

            opt_times.append(opt_time)
            conv_times.append(conv_time)

            opt_mean = np.mean(opt_times)
            opt_std = np.std(opt_times)
            conv_mean = np.mean(conv_times)
            conv_std = np.std(conv_times)

        print(
            f"N={N}, p={probability} | opt_mean={opt_mean:.4f}s, conv_mean={conv_mean:.4f}s")
        writer.writerow([N, probability, opt_mean,
                        opt_std, conv_mean, conv_std])
