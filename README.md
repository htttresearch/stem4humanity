# stem4humanity

**Mission:** 
Improve (the +STEM -> +Human Benefit) pipeline


**Step 0.0.1 — Math → Algorithms → Efficiency:** an *algorithm finder*. Classic
computer-science problems, classic solvers, and an open research question: how much
better can machine learning and AI approximate the hard ones? The long-term goal is a
platform where an AI can be told *"improve our nuclear fusion reactors' efficiency as
much as you can"* and has every letter of STEM at its disposal to get there. This step
gives it the **M**.

## Steps

| Step | Contents | Status |
|------|----------|--------|
| `steps/0-coml` | Prior TSP work (exact Held-Karp, Euclidean chained 2-opt, learned candidate ranker, C++ native) — kept as the reference and seed | archived reference |
| `algofinder` | The algorithm-finder engine: problem matrix, solver registry, benchmark harness, ML toolkit, LLM-discovery loop (later) | in development |

## The algorithm finder (algofinder/)

For each problem in the matrix (8-10 problems, 2-3 subproblems each):

- a **general** solver that works on the whole class,
- **specialized** solvers (preferably several) for each subproblem — going to a
  subproblem usually makes it easier,
- **exact** and **heuristic** solvers — and the research: how much better can we
  approximate with ML and AI, in several different ways,
- a **benchmark** of interesting instances with known or best-known solutions,
- **infrastructure** (harness, protocol, leaderboard, contribution docs) so other
  people can join the effort.

Everything is deterministic, reproducible, and machine-checked. See `algofinder/docs/`
for the protocol and how to add problems, solvers, and instances.
