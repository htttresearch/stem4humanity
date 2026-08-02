import random

import numpy as np

from stem4humanity.problems.bin_packing import BinPackingState
from stem4humanity.solvers.bin_packing.exact_dfs import BPDfsExactSolver


def brute_force(state):
    caps = state.capacities
    dims = state.dims
    items = list(range(state.item_count))
    sizes = [state.sizes[i] for i in items]
    order = sorted(items, key=lambda i: (-float(sizes[i].sum()), i))

    best = state.item_count

    def dfs(pos, loads):
        nonlocal best
        if len(loads) >= best:
            return
        if pos == len(order):
            best = min(best, len(loads))
            return
        size = sizes[order[pos]]
        tried = set()
        for i, load in enumerate(loads):
            key = tuple(round(float(x), 9) for x in load)
            if key in tried:
                continue
            tried.add(key)
            if np.all(load + size <= caps + 1e-9):
                loads[i] = load + size
                dfs(pos + 1, loads)
                loads[i] = load
        loads.append(size.copy())
        dfs(pos + 1, loads)
        loads.pop()

    dfs(0, [])
    return best


def random_instance(rng):
    dims = rng.randint(1, 3)
    n = rng.randint(4, 10)
    sizes = []
    for _ in range(n):
        item = [0.0] * dims
        for d in range(dims):
            if rng.random() < 0.85:
                item[d] = rng.uniform(0.08, 0.6)
            else:
                item[d] = rng.uniform(0.02, 0.06)
        sizes.append(item)
    return BinPackingState(f"fuzz", np.array(sizes), np.ones(dims))


def main():
    rng = random.Random(20260802)
    solver = BPDfsExactSolver(max_items=16)
    bad = 0
    for trial in range(1500):
        state = random_instance(rng)
        brute = brute_force(state)
        result = solver.solve(state)
        if result.cost != brute:
            bad += 1
            if bad <= 5:
                print("MISMATCH", trial, "solver", result.cost, "brute", brute)
                print("  sizes:", state.sizes.tolist())
                print("  solution:", result.solution)
    print(f"checked=1500 mismatches={bad}")


if __name__ == "__main__":
    main()
