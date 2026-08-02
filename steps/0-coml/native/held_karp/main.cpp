#include "held_karp.hpp"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace {

double peak_rss_mib() {
#if defined(__linux__) || defined(__APPLE__)
    struct rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) {
        return 0.0;
    }
#if defined(__APPLE__)
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(usage.ru_maxrss) / 1024.0;
#endif
#else
    return 0.0;
#endif
}

std::vector<double> read_distances(std::istream& input, int city_count) {
    std::vector<double> distances(
        static_cast<std::size_t>(city_count) * city_count
    );
    for (double& value : distances) {
        if (!(input >> value)) {
            throw std::runtime_error("failed to read distance matrix");
        }
    }
    return distances;
}

void print_result(
    const held_karp::SolveResult& result,
    double solve_seconds,
    double peak_memory_mib
) {
    std::cout << std::setprecision(17);
    std::cout << "cost=" << result.cost << '\n';
    std::cout << "states=" << result.states << '\n';
    std::cout << "solve_seconds=" << solve_seconds << '\n';
    std::cout << "peak_rss_mib=" << peak_memory_mib << '\n';
    std::cout << "tour=";
    for (std::size_t index = 0; index < result.tour.size(); ++index) {
        if (index > 0) {
            std::cout << ',';
        }
        std::cout << result.tour[index];
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::istream* input = &std::cin;
        std::ifstream file;

        for (int arg = 1; arg < argc; ++arg) {
            const std::string flag = argv[arg];
            if (flag == "--input" && arg + 1 < argc) {
                file.open(argv[++arg]);
                if (!file) {
                    throw std::runtime_error(
                        std::string("failed to open input file: ") + argv[arg]
                    );
                }
                input = &file;
            } else if (flag == "--help") {
                std::cout
                    << "Usage: held_karp_solver [--input distances.txt]\n"
                    << "Input format:\n"
                    << "  city_count\n"
                    << "  city_count * city_count row-major distances\n";
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + flag);
            }
        }

        int city_count = 0;
        if (!(*input >> city_count)) {
            throw std::runtime_error("failed to read city_count");
        }
        if (city_count < 1) {
            throw std::runtime_error("city_count must be positive");
        }

        const double rss_before = peak_rss_mib();
        const std::vector<double> distances =
            read_distances(*input, city_count);

        const auto started = std::chrono::steady_clock::now();
        const held_karp::SolveResult result =
            held_karp::solve(distances, city_count);
        const auto finished = std::chrono::steady_clock::now();

        const double solve_seconds =
            std::chrono::duration<double>(finished - started).count();
        const double peak_memory_mib = std::max(0.0, peak_rss_mib() - rss_before);

        print_result(result, solve_seconds, peak_memory_mib);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
