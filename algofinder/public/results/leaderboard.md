# Leaderboard

## bin-packing:large-items

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bp-exact-dfs | 4 | 0.00 | 4 | yes | 0.001 | 4 | 0 | 0 | 0 | 0 | 0 |
| bp-ffd | 8 | 0.00 | 8 | no | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| bp-large-items-exact | 8 | 0.00 | 8 | yes | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| bp-large-items-greedy | 8 | 0.00 | 8 | no | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| bp-learned-order | 8 | 3.71 | 4 | no | 0.046 | 0 | 0 | 0 | 0 | 0 | 0 |

## bin-packing:vector

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bp-exact-dfs | 8 | 0.00 | 8 | yes | 0.001 | 0 | 0 | 0 | 0 | 0 | 0 |
| bp-ffd | 8 | 0.00 | 8 | no | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| bp-learned-order | 8 | 7.32 | 3 | no | 0.121 | 0 | 0 | 0 | 0 | 0 | 0 |

## knapsack:0-1

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| knapsack-bnb-exact | 8 | 0.00 | 8 | yes | 0.070 | 0 | 0 | 0 | 0 | 0 | 0 |
| knapsack-dp-exact | 8 | 0.00 | 8 | yes | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| knapsack-greedy-density | 8 | 0.44 | 4 | no | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |

## knapsack:multidimensional

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| knapsack-bnb-exact | 6 | 0.00 | 6 | yes | 4.412 | 0 | 0 | 0 | 2 | 0 | 0 |
| knapsack-greedy-density | 8 | 2.78 | 0 | no | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |

## knapsack:subset-sum

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| knapsack-bnb-exact | 8 | 0.00 | 8 | yes | 0.033 | 0 | 0 | 0 | 0 | 0 | 0 |
| knapsack-dp-exact | 8 | 0.00 | 8 | yes | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| knapsack-greedy-density | 8 | 0.30 | 1 | no | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |

## parallel-scheduling:forest

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ps-exact | 1 | 0.00 | 1 | yes | 6.156 | 7 | 0 | 0 | 0 | 0 | 0 |
| ps-hu | 8 | 0.00 | 8 | yes | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| ps-cp-list | 8 | 16.19 | 1 | no | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| ps-list-scheduling | 8 | 16.19 | 1 | no | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| ps-learned-priority | 8 | 23.82 | 1 | no | 0.040 | 0 | 0 | 0 | 0 | 0 | 0 |

## parallel-scheduling:general

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ps-exact | 3 | 0.00 | 3 | yes | 4.339 | 5 | 0 | 0 | 0 | 0 | 0 |
| ps-cp-list | 8 | 12.78 | 0 | no | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| ps-list-scheduling | 8 | 18.89 | 0 | no | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| ps-learned-priority | 8 | 22.22 | 0 | no | 0.117 | 0 | 0 | 0 | 0 | 0 | 0 |

## scheduling:flow-shop-2

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| johnson-flow-shop | 12 | 0.00 | 12 | yes | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| learned-flow-shop | 12 | 0.00 | 12 | no | 0.005 | 0 | 0 | 0 | 0 | 0 | 0 |

## scheduling:job-shop

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| jobshop-bnb-exact | 12 | 0.00 | 12 | yes | 0.034 | 0 | 0 | 0 | 0 | 0 | 0 |
| jobshop-greedy | 12 | 27.92 | 1 | no | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |

## shortest-path:dag

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dag-topological-relaxation | 8 | 0.00 | 8 | yes | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| dijkstra | 8 | 0.00 | 8 | yes | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |
| learned-sp-pruning | 8 | 0.00 | 8 | yes | 0.051 | 0 | 0 | 0 | 0 | 0 | 0 |

## shortest-path:general

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dijkstra | 8 | 0.00 | 8 | yes | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |

## tsp:clustered

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chained-2opt-euclidean | 4 | - | 0 | no | 0.049 | 0 | 0 | 0 | 0 | 0 | 0 |
| distance-ranked-2opt | 4 | - | 0 | no | 0.035 | 0 | 0 | 0 | 0 | 0 | 0 |
| general-2opt | 4 | - | 0 | no | 0.114 | 0 | 0 | 0 | 0 | 0 | 0 |
| held-karp | 1 | - | 0 | yes | 3.010 | 3 | 0 | 0 | 0 | 0 | 0 |
| incremental-exact | 1 | - | 0 | yes | 26.894 | 3 | 0 | 0 | 0 | 0 | 0 |
| learned-candidate-2opt | 4 | - | 0 | no | 0.078 | 0 | 0 | 0 | 0 | 0 | 0 |

## tsp:euclidean

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| chained-2opt-euclidean | 20 | -0.00 | 6 | no | 0.042 | 0 | 0 | 0 | 0 | 0 | 0 |
| distance-ranked-2opt | 20 | -0.00 | 6 | no | 0.033 | 0 | 0 | 0 | 0 | 0 | 0 |
| learned-candidate-2opt | 20 | -0.00 | 5 | no | 0.170 | 0 | 0 | 0 | 0 | 0 | 0 |
| held-karp | 7 | -0.00 | 6 | yes | 0.431 | 13 | 0 | 0 | 0 | 0 | 0 |
| incremental-exact | 8 | 0.00 | 4 | yes | 0.016 | 9 | 0 | 0 | 3 | 0 | 0 |
| general-2opt | 20 | 0.01 | 4 | no | 0.104 | 0 | 0 | 0 | 0 | 0 | 0 |

## tsp:general

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| held-karp | 5 | -0.00 | 4 | yes | 0.641 | 3 | 0 | 0 | 0 | 0 | 0 |
| incremental-exact | 8 | -0.00 | 6 | yes | 0.026 | 0 | 0 | 0 | 0 | 0 | 0 |
| general-2opt | 8 | 4.82 | 1 | no | 0.003 | 0 | 0 | 0 | 0 | 0 | 0 |

## unit-commitment:classic

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uc-exact-dp | 8 | -0.00 | 7 | yes | 0.089 | 0 | 0 | 0 | 0 | 0 | 0 |
| uc-learned-commitment | 8 | 7.54 | 1 | no | 0.047 | 0 | 0 | 0 | 0 | 0 | 0 |
| uc-priority-list | 8 | 7.58 | 2 | no | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |

## unit-commitment:storage

| Solver | OK | Mean gap % | Best on | Exact | Mean s | Skipped | Unsupported | Error | Timeout | MemLim | Invalid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uc-storage-dp | 8 | 0.00 | 8 | yes | 0.034 | 0 | 0 | 0 | 0 | 0 | 0 |
| uc-storage-learned | 8 | 0.85 | 0 | no | 0.043 | 0 | 0 | 0 | 0 | 0 | 0 |
| uc-storage-threshold | 8 | 5.67 | 0 | no | 0.000 | 0 | 0 | 0 | 0 | 0 | 0 |

## Summary

- problems with runs: 16
- solver cells: 496
- ok: 444
- skipped: 47
- unsupported: 0
- error: 0
- invalid: 0
- timeout: 5
- memory_limit: 0

