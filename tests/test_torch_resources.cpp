#include "ezlocalai/ctts_torch.hpp"
#include <ATen/ATen.h>
#include <iostream>
#include <stdexcept>
namespace {
void check(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
}
int main(int argc, char **argv) {
    try {
        const bool require_cuda = argc > 1 && std::string(argv[1]) == "cuda";
        auto probes = ezlocalai::torch_resource_probes();
        auto operations = ezlocalai::ctts_torch_operations();
        if (!operations.cuda_available()) {
            check(!require_cuda, "CUDA required for this run");
            check(probes.allocator_stats().empty(), "Unavailable CUDA must produce no allocator rows");
            probes.release_unused_cuda_memory();
            std::cout << "Torch without CUDA: empty diagnostics and safe cleanup\n"; return 0;
        }
        probes.release_unused_cuda_memory();
        const auto before = probes.allocator_stats(); check(!before.empty(), "Missing CUDA device");
        for (const auto &device : before) {
            check(device.error.empty(), "Initial CUDA stats failed");
            const auto id = device.index;
            auto tensor = at::empty({64 * 1024 * 1024}, at::TensorOptions().dtype(at::kByte).device(at::Device(at::kCUDA, id)));
            const auto allocated = probes.allocator_stats().at(id);
            check(allocated.error.empty(), "Allocated CUDA stats failed");
            check(allocated.allocated_gb >= device.allocated_gb + 0.061, "Live allocation missing from counters");
            check(allocated.reserved_gb >= allocated.allocated_gb, "Reserved allocation too small");
            check(allocated.max_allocated_gb >= allocated.allocated_gb, "Peak allocation too small");
            tensor = at::Tensor();
            const auto released = probes.allocator_stats().at(id);
            check(released.allocated_gb == device.allocated_gb, "Released tensor remains allocated");
            check(released.reserved_gb >= allocated.reserved_gb, "Released block should remain cached");
            check(released.max_allocated_gb == allocated.max_allocated_gb, "Peak lost after release");
            probes.release_unused_cuda_memory();
            const auto cleared = probes.allocator_stats().at(id);
            check(cleared.reserved_gb < released.reserved_gb, "Cache cleanup did not release reservation");
            check(cleared.max_allocated_gb == allocated.max_allocated_gb, "Cache cleanup reset peak");
            std::cout << "GPU " << id << ": allocated=" << allocated.allocated_gb << " reserved=" << released.reserved_gb << " cleared=" << cleared.reserved_gb << " peak=" << cleared.max_allocated_gb << " GiB\n";
        }
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
