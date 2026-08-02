# TSPLIB EUC_2D benchmark (5s budget, certified optima)

| Solver | tsp:tsplib:berlin52 | tsp:tsplib:ch150 | tsp:tsplib:eil51 | tsp:tsplib:eil76 | tsp:tsplib:kroA200 | tsp:tsplib:lin318 | tsp:tsplib:pr299 | tsp:tsplib:u574 | Mean gap % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| strong-ils-euclidean |  0.00 |  0.58 |  0.23 |  1.12 |  0.60 |  3.11 |  3.13 |  5.22 |  1.75 |
| chained-2opt-euclidean |  0.00 |  3.71 |  2.11 |  4.65 |  2.70 |  3.80 |  2.81 |  5.54 |  3.16 |
| distance-ranked-2opt |  0.82 |  5.78 |  1.17 |  4.09 |  2.61 |  3.67 |  2.64 |  5.12 |  3.24 |
| learned-candidate-2opt |  0.82 |  5.84 |  2.11 |  4.83 |  2.62 |  3.85 |  2.81 |  5.86 |  3.59 |
| general-2opt | 10.41 |  4.66 |  2.58 |  6.51 | 11.13 |  9.86 | 12.83 |   -   |  8.28 |

## Per-instance integer costs and opt

| Instance | n | Opt | chained-2opt-euclidean | distance-ranked-2opt | general-2opt | learned-candidate-2opt | strong-ils-euclidean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| tsp:tsplib:berlin52 | 52 | 7542 | 7542 | 7604 | 8327 | 7604 | 7542 |
| tsp:tsplib:ch150 | 150 | 6528 | 6770 | 6905 | 6832 | 6909 | 6566 |
| tsp:tsplib:eil51 | 51 | 426 | 435 | 431 | 437 | 435 | 427 |
| tsp:tsplib:eil76 | 76 | 538 | 563 | 560 | 573 | 564 | 544 |
| tsp:tsplib:kroA200 | 200 | 29368 | 30160 | 30134 | 32637 | 30136 | 29544 |
| tsp:tsplib:lin318 | 318 | 42029 | 43626 | 43573 | 46173 | 43649 | 43336 |
| tsp:tsplib:pr299 | 299 | 48191 | 49544 | 49462 | 54373 | 49544 | 49699 |
| tsp:tsplib:u574 | 574 | 36905 | 38949 | 38793 | - | 39068 | 38833 |

## Wall seconds (mean per solver)

- chained-2opt-euclidean: 3.06s
- distance-ranked-2opt: 2.81s
- general-2opt: 4.13s
- learned-candidate-2opt: 3.02s
- strong-ils-euclidean: 2.92s
