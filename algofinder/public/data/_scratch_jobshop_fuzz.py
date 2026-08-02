import itertools
import random

import numpy as np

from stem4humanity.problems.scheduling import JobShopState
from stem4humanity.solvers.scheduling.jobshop_bnb import JobShopBnBExactSolver, simulate_makespan


def correct_makespan(state, orders):
    """Ground-truth earliest-start makespan; inf for infeasible orderings.

    Evaluates in topological order over machine-queue and job-precedence
    dependencies, so evaluation order is independent of machine numbering.
    """
    position = {}
    for machine, entries in orders.items():
        for idx, (job, step) in enumerate(entries):
            position[(job, step)] = (machine, idx)
    n_nodes = sum(len(entries) for entries in orders.values())
    if n_nodes == 0:
        return 0.0
    adj = {}
    indeg = {}
    for machine, entries in orders.items():
        for idx in range(len(entries) - 1):
            a = (machine, idx)
            b = (machine, idx + 1)
            adj.setdefault(a, []).append(b)
            indeg[b] = indeg.get(b, 0) + 1
        for idx, (job, step) in enumerate(entries):
            if step + 1 < len(state.operations[job]):
                nxt = position[(job, step + 1)]
                a = (machine, idx)
                adj.setdefault(a, []).append(nxt)
                indeg[nxt] = indeg.get(nxt, 0) + 1
    ready = [n for n in ((m, i) for m, entries in orders.items() for i in range(len(entries))) if indeg.get(n, 0) == 0]
    topo = []
    while ready:
        node = ready.pop()
        topo.append(node)
        for nxt in adj.get(node, []):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
    if len(topo) < n_nodes:
        return float("inf")
    machine_free = [0.0] * state.n_machines
    job_finish = [[0.0] * (len(ops) + 1) for ops in state.operations]
    makespan = 0.0
    for machine, idx in topo:
        job, step = orders[machine][idx]
        duration = state.operations[job][step][1]
        start = max(machine_free[machine], job_finish[job][step])
        finish = start + duration
        job_finish[job][step + 1] = finish
        machine_free[machine] = finish
        makespan = max(makespan, finish)
    return makespan


def brute_force(state):
    per_machine = {}
    for machine in state.machines:
        per_machine[machine] = [
            (job, step)
            for job in state.jobs
            for step, (op_m, _) in enumerate(state.operations[job])
            if op_m == machine
        ]
    best = float("inf")
    for combination in itertools.product(*(itertools.permutations(per_machine[m]) for m in state.machines)):
        orders = {m: list(o) for m, o in zip(state.machines, combination)}
        cost = correct_makespan(state, orders)
        best = min(best, cost)
    return best


def random_shop(rng, n_jobs, n_machines, max_len=3, lo=1, hi=6):
    ops = []
    for _ in range(n_jobs):
        length = rng.randint(1, max_len)
        steps = []
        for _ in range(length):
            machine = rng.randrange(n_machines)
            duration = rng.randint(lo, hi)
            steps.append((machine, float(duration)))
        ops.append(steps)
    return JobShopState("fuzz", n_jobs, n_machines, ops)


def main():
    rng = random.Random(20260801)
    solver = JobShopBnBExactSolver()
    bad = 0
    checked = 0
    for trial in range(3000):
        n_jobs = rng.randint(2, 3)
        n_machines = rng.randint(2, 3)
        state = random_shop(rng, n_jobs, n_machines)
        brute = brute_force(state)
        result = solver.solve(state)
        checked += 1
        if result.cost != brute:
            bad += 1
            if bad <= 5:
                print("MISMATCH", trial, "solver", result.cost, "brute", brute)
                print("  ops:", state.operations)
                print("  solution:", result.solution)
    print(f"checked={checked} mismatches={bad}")


if __name__ == "__main__":
    main()
