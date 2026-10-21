#include <atomic>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

// Link the engine implementation itself. Each invocation starts with an
// uninitialized conversion table, as parallel model loading does.
std::string convert_diffusers_vae_to_original_sd1(std::string name);

int main() {
    constexpr int count = 32;
    std::atomic<int> ready{0}, errors{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    for (int i = 0; i < count; ++i) {
        threads.emplace_back([&] {
            ++ready;
            while (!start.load()) std::this_thread::yield();
            for (int repeat = 0; repeat < 100; ++repeat) {
                for (int block = 0; block < 4; ++block) {
                    const auto input = "decoder.up_blocks." + std::to_string(block) + ".resnets.2.conv1.weight";
                    const auto expected = "decoder.up." + std::to_string(3 - block) + ".block.2.conv1.weight";
                    if (convert_diffusers_vae_to_original_sd1(input) != expected) ++errors;
                }
                if (convert_diffusers_vae_to_original_sd1("encoder.mid_block.resnets.1.conv1.weight")
                    != "encoder.mid.block_2.conv1.weight") ++errors;
            }
        });
    }
    while (ready.load() != count) std::this_thread::yield();
    start = true;
    for (auto &thread : threads) thread.join();
    if (errors.load()) {
        std::cerr << errors << " incomplete or incorrect concurrent mappings\n";
        return 1;
    }
    std::cout << "Verified 16000 mappings across 32 concurrent first-use threads\n";
}
