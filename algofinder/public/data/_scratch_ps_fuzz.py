import random

import numpy as np

from algofinder.problems.parallel_scheduling import ParallelSchedulingState
from algofinder.solvers.parallel_scheduling.exact import PSExactSolver


def brute_force(state):
    best = float("inf")
    best_solution = None
    machine_free = np.zeros(state.machines, float)
    start = [0.0] * state.task_count
    assignment = [-1] * state.task_count
    ready = [t for t in state.tasks if not state.predecessors[t]]
    unscheduled_preds = [len(state.predecessors[t]) for t in state.tasks]
    used = 0

    def dfs(makespan):
        nonlocal best, best_solution, used
        if not ready:
            if makespan < best:
                best = makespan
                best_solution = [[float(assignment[t]), start[t]] for t in state.tasks]
            return
        if makespan >= best:
            return
        for position in range(len(ready)):
            task = ready.pop(position)
            pred_finish = 0.0
            for p in state.predecessors[task]:
                pred_finish = max(pred_finish, start[p] + float(state.times[p]))
            for s in state.successors[task]:
                unscheduled_preds[s] -= 1
                if unscheduled_preds[s] == 0:
                    ready.append(s)
            tried = set()
            for machine in range(min(used + 1, state.machines)):
                free = float(machine_free[machine])
                if free in tried:
                    continue
                tried.add(free)
                s = max(pred_finish, free)
                finish = s + float(state.times[task])
                old_free = free
                old_used = used
                machine_free[machine] = finish
                assignment[task] = machine
                start[task] = s
                used = max(used, machine + 1)
                dfs(max(makespan, finish))
                machine_free[machine] = old_free
                used = old_used
            for s in state.successors[task]:
                if unscheduled_preds[s] == 0:
                    ready.pop()
                unscheduled_preds[s] += 1
            ready.insert(position, task)

    dfs(0.0)
    return best, best_solution


def random_instance(rng):
    n = rng.randint(2, 9)
    m = rng.randint(1, 3)
    times = [float(rng.randint(1, 8)) for _ in range(n)]
    preds = [[] for _ in range(n)]
    for a in range(n):
        for b in range(a + 1, n):
            if rng.random() < 0.18:
                preds[b].append(a)
    return ParallelSchedulingState("fuzz", n, m, times, preds)


def main():
    rng = random.Random(20260804)
    solver = PSExactSolver(max_tasks=12, deadline_seconds=30.0)
    bad = 0
    max_nodes = 0
    for trial in range(1000):
        state = random_instance(rng)
        brute, brute_sol = brute_force(state)
        result = solver.solve(state)
        max_nodes = max(max_nodes, result.metadata["nodes"])
        if result.cost != brute or not result.exact:
            bad += 1
            if bad <= 5:
                print("MISMATCH", trial, "solver", result.cost, result.exact, "brute", brute)
                print("  times:", state.times.tolist(), "m:", state.machines)
                print("  preds:", list(state.predecessors))
    print(f"checked=1000 mismatches={bad} max_nodes={max_nodes}")


if __name__ == "__main__":
    main()
