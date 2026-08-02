#include "held_karp.hpp"

#include <cstdint>
#include <stdexcept>

namespace held_karp {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

double distance_at(
    const std::vector<double>& distances,
    int city_count,
    int row,
    int column
) {
    return distances[static_cast<std::size_t>(row) * city_count + column];
}

}  // namespace

SolveResult solve(const std::vector<double>& distances, int city_count) {
    if (city_count < 1) {
        throw std::invalid_argument("city_count must be positive");
    }

    const std::size_t expected =
        static_cast<std::size_t>(city_count) * city_count;
    if (distances.size() != expected) {
        throw std::invalid_argument("distance matrix has unexpected size");
    }

    SolveResult result;

    if (city_count == 1) {
        result.cost = 0.0;
        result.tour = {0};
        result.states = 1;
        return result;
    }

    if (city_count == 2) {
        result.cost =
            distance_at(distances, city_count, 0, 1)
            + distance_at(distances, city_count, 1, 0);
        result.tour = {0, 1};
        result.states = 2;
        return result;
    }

    const int non_start_count = city_count - 1;
    const int subset_count = 1 << non_start_count;
    const int full_subset = subset_count - 1;

    std::vector<double> costs(
        static_cast<std::size_t>(subset_count) * non_start_count
    );
    std::vector<std::uint8_t> parents(
        static_cast<std::size_t>(subset_count) * non_start_count
    );
    std::vector<std::uint8_t> bit_to_index(subset_count);

    auto cost_at = [&](int subset, int endpoint) -> double& {
        return costs[static_cast<std::size_t>(subset) * non_start_count + endpoint];
    };
    auto parent_at = [&](int subset, int endpoint) -> std::uint8_t& {
        return parents[static_cast<std::size_t>(subset) * non_start_count + endpoint];
    };

    for (int city = 0; city < non_start_count; ++city) {
        const int bit = 1 << city;
        bit_to_index[bit] = static_cast<std::uint8_t>(city);
        cost_at(bit, city) = distance_at(distances, city_count, 0, city + 1);
    }

    for (int subset = 1; subset < subset_count; ++subset) {
        int endpoint_bits = subset;

        while (endpoint_bits != 0) {
            const int endpoint_bit = endpoint_bits & -endpoint_bits;
            const int endpoint = bit_to_index[endpoint_bit];
            const int previous_subset = subset ^ endpoint_bit;

            if (previous_subset != 0) {
                double best_cost = kInfinity;
                int best_parent = 0;
                int predecessor_bits = previous_subset;

                while (predecessor_bits != 0) {
                    const int predecessor_bit = predecessor_bits & -predecessor_bits;
                    const int predecessor = bit_to_index[predecessor_bit];

                    const double candidate =
                        cost_at(previous_subset, predecessor)
                        + distance_at(
                            distances,
                            city_count,
                            predecessor + 1,
                            endpoint + 1
                        );

                    if (candidate < best_cost) {
                        best_cost = candidate;
                        best_parent = predecessor;
                    }

                    predecessor_bits ^= predecessor_bit;
                }

                cost_at(subset, endpoint) = best_cost;
                parent_at(subset, endpoint) =
                    static_cast<std::uint8_t>(best_parent);
            }

            endpoint_bits ^= endpoint_bit;
        }
    }

    double best_tour_cost = kInfinity;
    int final_city = -1;

    for (int city = 0; city < non_start_count; ++city) {
        const double candidate =
            cost_at(full_subset, city)
            + distance_at(distances, city_count, city + 1, 0);

        if (candidate < best_tour_cost) {
            best_tour_cost = candidate;
            final_city = city;
        }
    }

    if (final_city < 0 || best_tour_cost == kInfinity) {
        throw std::runtime_error("no finite tour found");
    }

    result.cost = best_tour_cost;
    result.tour.assign(city_count, 0);
    result.tour[0] = 0;

    int subset = full_subset;
    int current = final_city;

    for (int position = city_count - 1; position >= 1; --position) {
        result.tour[position] = current + 1;

        if (position > 1) {
            const int previous = parent_at(subset, current);
            subset ^= 1 << current;
            current = previous;
        }
    }

    result.states = static_cast<std::size_t>(
        1 + non_start_count * (1 << (non_start_count - 1))
    );
    return result;
}

}  // namespace held_karp
