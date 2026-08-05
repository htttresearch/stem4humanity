# Portfolio report

- instances with ok runs: 144 (ok runs: 444)
- splits present in runs: test
- virtual-best instances: 144

## Virtual best vs solvers

| Solver | Instances | Wins | Mean regret % | Median regret % | Max regret % |
|---|---|---|---|---|---|
| bp-exact-dfs | 12 | 12 | 0.0 | 0.0 | 0.0 |
| bp-ffd | 16 | 16 | 0.0 | 0.0 | 0.0 |
| bp-large-items-exact | 8 | 8 | 0.0 | 0.0 | 0.0 |
| bp-large-items-greedy | 8 | 8 | 0.0 | 0.0 | 0.0 |
| dag-topological-relaxation | 8 | 8 | 0.0 | 0.0 | 0.0 |
| dijkstra | 16 | 16 | 0.0 | 0.0 | 0.0 |
| held-karp | 13 | 9 | 0.0 | 0.0 | 0.0 |
| incremental-exact | 17 | 8 | 0.0 | 0.0 | 0.0 |
| jobshop-bnb-exact | 12 | 12 | 0.0 | 0.0 | 0.0 |
| johnson-flow-shop | 12 | 12 | 0.0 | 0.0 | 0.0 |
| knapsack-greedy-density | 24 | 24 | 0.0 | 0.0 | 0.0 |
| learned-flow-shop | 12 | 12 | 0.0 | 0.0 | 0.0 |
| learned-sp-pruning | 8 | 8 | 0.0 | 0.0 | 0.0 |
| ps-exact | 4 | 4 | 0.0 | 0.0 | 0.0 |
| ps-hu | 8 | 8 | 0.0 | 0.0 | 0.0 |
| uc-exact-dp | 8 | 8 | 0.0 | 0.0 | 0.0 |
| uc-storage-dp | 8 | 8 | 0.0 | 0.0 | 0.0 |
| learned-candidate-2opt | 24 | 16 | 0.0759 | 0.0 | 1.3593 |
| chained-2opt-euclidean | 24 | 17 | 0.0944 | 0.0 | 0.8889 |
| distance-ranked-2opt | 24 | 18 | 0.2082 | 0.0 | 2.8942 |
| knapsack-dp-exact | 16 | 5 | 0.3725 | 0.2118 | 1.4622 |
| uc-storage-learned | 8 | 0 | 0.8491 | 0.729 | 1.3961 |
| knapsack-bnb-exact | 22 | 5 | 1.0652 | 0.4141 | 8.1119 |
| general-2opt | 32 | 3 | 2.934 | 1.8469 | 10.7202 |
| bp-learned-order | 16 | 7 | 5.5161 | 6.5126 | 18.1818 |
| uc-storage-threshold | 8 | 0 | 5.6729 | 5.421 | 9.21 |
| uc-learned-commitment | 8 | 1 | 7.5412 | 6.1149 | 19.5402 |
| uc-priority-list | 8 | 2 | 7.5793 | 6.2881 | 19.5402 |
| ps-cp-list | 16 | 6 | 10.4911 | 12.1429 | 25.0 |
| ps-list-scheduling | 16 | 1 | 15.3438 | 16.6667 | 25.0 |
| ps-learned-priority | 16 | 2 | 20.0268 | 17.1569 | 50.0 |
| jobshop-greedy | 12 | 1 | 27.9199 | 29.5139 | 64.1026 |

## Feature cost

| Feature set | Records | Mean timing s | Median timing s | Mean peak RSS B |
|---|---|---|---|---|
| etsp-geometry@1 | 36 | 0.002824 | 0.00145 | 52490695 |
| params@1 | 212 | 1.5e-05 | 1.4e-05 | 50935692 |

## Per-split gap vs virtual best

- test: 444 ok runs, mean gap 3.2941%
