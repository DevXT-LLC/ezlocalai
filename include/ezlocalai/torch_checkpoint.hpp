#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace ezlocalai {
// Metadata from a weights-only torch.save archive. Storage bytes remain owned
// by the archive reader; no Python objects or global functions are executed.
struct TorchCheckpointTensor {
    std::string storage_key;
    std::string scalar_type;
    int64_t storage_elements = 0;
    int64_t storage_offset = 0;
    std::vector<int64_t> shape, strides;
};
std::map<std::string, TorchCheckpointTensor> parse_torch_checkpoint_pickle(std::string_view pickle);
}
