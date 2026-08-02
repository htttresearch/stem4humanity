implemen, for each "area/domain", 1 problem in 2 versions: 1 general and one subclass.

| Family                   | General Problem                  | Well-Studied Subclass                           | Specialized Solution                  | Nature of Improvement                     |
|--------------------------|----------------------------------|-------------------------------------------------|---------------------------------------|--------------------------------------------|
| Ordering                 | Sorting or top-k selection       | Bounded-universe integer sorting                | Counting/radix sort                   | Exact asymptotic improvement               |
| Graphs                   | Shortest path                    | Shortest paths in DAGs                          | Topological-order relaxation           | Exact asymptotic improvement               |
| Combinatorial optimization| Traveling salesperson (TSP)     | 2D Euclidean TSP                               | Geometric PTAS                        | Better approximation guarantee             |
| Mathematical programming | Small linear/integer programs    | Minimum-cost network flow                       | Cost scaling/network simplex           | Structural and practical improvement       |
| Algebra                  | Polynomial multiplication        | Sparse polynomial multiplication                | Output-sensitive sparse multiplication | Improvement depends on sparsity, not degree|
| Strings                  | Edit distance                    | Small-edit-distance instances                   | Ukkonen/banded algorithms              | Parameterized exact improvement            |
| Dynamic programming      | Matrix-chain multiplication      | Monotone dimension sequences                    | One-sided greedy parenthesization      | Exact simplification                       |
| Search                   | SAT                              | Horn-SAT                                        | Forward chaining/propagation           | Exponential-to-linear transition           |
| Packing                  | Bin packing                      | All items larger than 1/3                       | Maximum matching                       | NP-hard-to-polynomial transition           |
| Scheduling               | Single/job-shop scheduling       | Two-machine flow shop, makespan                  | Johnson’s rule                        | NP-hard-to-O(n log n) transition           |

then, for each of these domains, implement 3 solvers:
1. a classic general solving algorithm for the whole class
2. a highly specialized but manually found heuristics for the particular subclass we're considering
3. a ML infused to also try to solve the particular subclass, but instead of manually found heuristics, ML learned heuristics

create a very thorough "running env" where each algo is run against instances of the problem, both general and in the particular subclass

evalaute

