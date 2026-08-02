import itertools
import random

import numpy as np

from stem4humanity.problems.knapsack import KnapsackState
from stem4humanity.solvers.knapsack.branch_and_bound import BranchAndBoundExact


def brute_force(state):
    best = 0.0
    best_set = ()
    for mask in range(1 << state.item_count):
        selected = [i for i in range(state.item_count) if mask >> i & 1]
        if not selected:
            continue
        total_w = state.weights[selected].sum(axis=0)
        if np.all(total_w <= state.capacities + 1e-9):
            value = float(state.values[selected].sum())
            if value > best:
                best = value
                best_set = tuple(selected)
    return best


def random_instance(rng):
    dims = rng.choice([1, 1, 2])
    n = rng.randint(6, 16)
    weights = []
    values = []
    capacities = []
    for _ in range(n):
        row = []
        for _ in range(dims):
            row.append(rng.randint(1, 30))
        weights.append(row)
        values.append(rng.randint(1, 40))
    for d in range(dims):
        capacities.append(rng.randint(10, n * 30 // 3))
    return KnapsackState("fuzz", np.array(weights, float), np.array(values, float), np.array(capacities, float))


def main():
    rng = random.Random(20260803)
    solver = BranchAndBoundExact()
    bad = 0
    max_nodes = 0
    for trial in range(1200):
        state = random_instance(rng)
        brute = brute_force(state)
        result = solver.solve(state)
        max_nodes = max(max_nodes, result.metadata["nodes"])
        if result.cost != brute or not result.exact:
            bad += 1
            if bad <= 5:
                print("MISMATCH", trial, "solver", result.cost, result.exact, "brute", brute)
                print("  weights:", state.weights.tolist())
                print("  values:", state.values.tolist())
                print("  caps:", state.capacities.tolist())
    print(f"checked=1200 mismatches={bad} max_nodes={max_nodes}")


if __name__ == "__main__":
    main()
