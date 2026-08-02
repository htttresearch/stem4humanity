import random

from algofinder.problems.shortest_path import ShortestPathState
from algofinder.solvers.shortest_path.dijkstra import DijkstraSolver
from algofinder.solvers.shortest_path.learned_pruner import LearnedArcPruningSolver


def random_dag(rng):
    n = rng.randint(3, 9)
    source = 0
    arcs = []
    for tail in range(n):
        for head in range(tail + 1, n):
            if rng.random() < 0.45:
                parallel = rng.randint(1, 3)
                for _ in range(parallel):
                    weight = rng.randint(1, 12)
                    arcs.append([tail, head, float(weight)])
    if not arcs:
        arcs = [[0, n - 1, 1.0]]
    return ShortestPathState("fuzz", n, source, arcs)


def main():
    rng = random.Random(20260805)
    pruner = LearnedArcPruningSolver(threshold=0.35)
    dijkstra = DijkstraSolver()
    bad = 0
    fallbacks = 0
    for trial in range(1500):
        state = random_dag(rng)
        result = pruner.solve(state)
        reference = dijkstra.solve(state)
        if not result.exact:
            bad += 1
            if bad <= 3:
                print("NOT EXACT", trial, result.metadata)
            continue
        if result.cost != reference.cost:
            bad += 1
            if bad <= 5:
                print("MISMATCH", trial, "pruner", result.cost, "dijkstra", reference.cost)
                print("  arcs:", state.arcs.tolist())
        if result.metadata["fallback_used"]:
            fallbacks += 1
    print(f"checked=1500 mismatches={bad} fallbacks={fallbacks}")


if __name__ == "__main__":
    main()
