#include "runtime/denoiser.hpp"
#include "ezlocalai/ltx_runtime.hpp"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("fixture required");
        std::ifstream input(argv[1]);
        nlohmann::json fixture;
        input >> fixture;
        float worst = 0;
        size_t baseline_mismatches = 0;
        for (const auto &row : fixture.at("cases")) {
            const int steps = row.at("steps"), tokens = row.at("actual_tokens");
            const auto expected = row.at("sigmas").get<std::vector<float>>();
            LTX2Scheduler corrected(tokens, ezlocalai::ltx_reference_scheduler_args.data());
            const auto actual = corrected.get_sigmas(steps, 0, 1, {});
            if (actual.size() != expected.size()) throw std::runtime_error("schedule shape");
            for (size_t i = 0; i < expected.size(); ++i) {
                const float error = std::abs(actual[i] - expected[i]);
                worst = std::max(worst, error);
                if (!std::isfinite(actual[i]) || error > fixture.at("absolute_tolerance").get<float>())
                    throw std::runtime_error("installed scheduler mismatch");
            }
            // Ensure these cases expose the original size-dependent schedule.
            const auto baseline = LTX2Scheduler(tokens).get_sigmas(steps, 0, 1, {});
            if (std::abs(baseline[1] - expected[1]) > 1e-5f) ++baseline_mismatches;
        }
        if (fixture.at("cases").size() != 72 || baseline_mismatches == 0)
            throw std::runtime_error("all source cases and baseline regression required");
        std::cout << "72 actual pipeline schedules match; max sigma error " << worst
                  << "; baseline fails " << baseline_mismatches << " cases\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
