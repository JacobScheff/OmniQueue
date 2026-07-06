#pragma once

#include <cstdint>
#include <random>
#include <string>

#include <torch/script.h>

namespace park {

class PolicyRunner {
public:
    PolicyRunner(const std::string& torchscript_path, const std::string& device = "cpu");

    bool is_available() const { return loaded_; }

    void infer_batch(
        const float* obs_flat,
        int batch_size,
        int64_t* actions_out,
        float* logprobs_out,
        float* values_out,
        bool stochastic,
        std::mt19937_64& rng);

private:
    bool loaded_ = false;
    torch::jit::script::Module module_;
    torch::Device device_{torch::kCPU};
};

bool libtorch_enabled();

}  // namespace park
