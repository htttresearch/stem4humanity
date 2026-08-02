#pragma once

#include <cstddef>
#include <limits>
#include <vector>

namespace held_karp {

struct SolveResult {
    double cost = std::numeric_limits<double>::infinity();
    std::vector<int> tour;
    std::size_t states = 0;
};

/// Exact Held--Karp DP with city zero fixed as the start.
///
/// ``distances`` is a row-major ``city_count x city_count`` matrix.
SolveResult solve(const std::vector<double>& distances, int city_count);

}  // namespace held_karp
