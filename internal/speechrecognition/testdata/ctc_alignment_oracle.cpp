// Probe driver only. The implementation is compiled from the pinned audio.cpp checkout.
#include "engine/community_models/mms_forced_aligner/ctc_alignment.h"
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

int main() {
    using namespace engine::community_models::mms_forced_aligner;
    int frames, classes, blank, count;
    while (std::cin >> frames >> classes >> blank >> count) {
        std::vector<int32_t> targets(count);
        std::vector<float> probabilities(frames * classes);
        for (auto &value : targets) std::cin >> value;
        for (auto &value : probabilities) std::cin >> value;
        const auto result = ctc_forced_align({probabilities.data(), frames, classes}, targets, blank,
            {int64_t(frames) * (2 * count + 1), count});
        std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
        std::cout << "{\"states\":[";
        for (size_t i = 0; i < result.state_path.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << result.state_path[i];
        }
        std::cout << "],\"score\":" << result.score << "}\n";
    }
}
