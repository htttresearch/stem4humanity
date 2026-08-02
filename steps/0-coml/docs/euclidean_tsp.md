# 2D Euclidean TSP study

This implementation treats a 2D Euclidean TSP as a complete symmetric graph
derived from city coordinates. A tour is a permutation of city indices; its
return edge is implicit.

## Solvers

- `HeldKarpSolver` is the classic exact general solver. It accepts every
  symmetric complete distance matrix, but is intentionally limited to small
  instances because its cost is `O(n² 2ⁿ)`.
- `EuclideanChainedTwoOptSolver` is the hand-designed planar specialization:
  convex-hull regret insertion, k-nearest plus directional candidate edges,
  candidate-restricted 2-opt, and double-bridge restarts.
- `LearnedCandidateTwoOptSolver` retains the same feasible construction and
  local-search operations but uses a trained random-forest ranker to retain
  candidate edges. The model predicts whether an edge belongs to an exact
  reference tour; it never directly emits a tour.

The generator mixes uniform-square, Gaussian-clustered, and noisy-corridor
families. The training and test splits are generated as separate instances, so
candidate edges from one instance cannot leak into the other split.

## Quick run

Use the project virtual environment from the repository root:

```bash
coml/bin/python -m unittest discover -s tests -v

coml/bin/python -m experiments.generate_euclidean_tsp \
  --output artifacts/data/euclidean.json --instances 30 --city-count 16 --seed 7

coml/bin/python -m experiments.train_euclidean_tsp \
  --output artifacts/models/candidate_ranker.joblib \
  --instances 60 --validation-instances 18 --city-count 14 --seed 7

coml/bin/python -m experiments.benchmark_euclidean_tsp \
  --model artifacts/models/candidate_ranker.joblib \
  --output artifacts/results/benchmark.csv \
  --city-counts 10 25 50 --instances 6 --seed 7
```

The benchmark records the exact solver where feasible, the manual solver, the
learned hybrid, and a distance-ranked candidate control with the same `top_k`
budget as the learned model. It reports solution cost, gap to the exact or
manual reference, candidate-edge recall, wall time, and available
preprocessing/model/search counters.

All pipeline commands print detailed progress by default. Pass `--quiet` to
suppress status lines (sklearn may still print tree-building progress during
training).

## Interpretation

Treat a learned improvement as supported only if the learned candidate ranking
beats the distance-ranked control under the same candidate budget and solver
settings. Candidate recall alone is not sufficient: inspect its effect on
solution quality, move evaluations, and end-to-end time.
