#include "policy_runner.hpp"

#include <array>
#include <cmath>
#include <random>
#include <stdexcept>

#include <torch/torch.h>

#include "park_sim.hpp"

namespace park {

namespace {

torch::Device resolve_device(const std::string& device_name) {
    if (device_name == "cuda" && torch::cuda::is_available()) {
        return torch::Device(torch::kCUDA);
    }
    return torch::Device(torch::kCPU);
}

}  // namespace

bool libtorch_enabled() {
    return true;
}

PolicyRunner::PolicyRunner(const std::string& torchscript_path, const std::string& device)
    : device_(resolve_device(device)) {
    try {
        module_ = torch::jit::load(torchscript_path, device_);
        module_.eval();
        loaded_ = true;
    } catch (const c10::Error& ex) {
        throw std::runtime_error(
            std::string("Failed to load TorchScript policy: ") + ex.what() +
            " (path=" + torchscript_path + ")");
    }
}

void PolicyRunner::infer_batch(
    const float* obs_flat,
    int batch_size,
    int64_t* actions_out,
    float* logprobs_out,
    float* values_out,
    bool stochastic,
    std::mt19937_64& rng) {
    if (!loaded_) {
        throw std::runtime_error("PolicyRunner module is not loaded.");
    }
    if (batch_size <= 0) {
        return;
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device_);
    torch::Tensor input =
        torch::from_blob(const_cast<float*>(obs_flat), {batch_size, kFlatObsDim}, options).clone();

    torch::NoGradGuard no_grad;
    auto output = module_.forward({input});
    torch::Tensor logits;
    torch::Tensor values;

    if (output.isTuple()) {
        auto tuple = output.toTuple();
        logits = tuple->elements()[0].toTensor();
        values = tuple->elements()[1].toTensor();
    } else {
        throw std::runtime_error("TorchScript policy must return (logits, value) tuple.");
    }

    logits = logits.to(torch::kCPU);
    values = values.to(torch::kCPU).view({batch_size});
    const torch::Tensor log_softmax = torch::log_softmax(logits, /*dim=*/1);

    if (stochastic) {
        const torch::Tensor probs = torch::softmax(logits, /*dim=*/1);
        auto probs_a = probs.accessor<float, 2>();
        auto log_soft_a = log_softmax.accessor<float, 2>();
        auto values_a = values.accessor<float, 1>();
        for (int i = 0; i < batch_size; ++i) {
            std::array<double, kNumActions> weights{};
            for (int a = 0; a < kNumActions; ++a) {
                weights[static_cast<size_t>(a)] =
                    std::max(0.0, static_cast<double>(probs_a[i][a]));
            }
            std::discrete_distribution<int> dist(weights.begin(), weights.end());
            const int action = dist(rng);
            actions_out[i] = action;
            logprobs_out[i] = log_soft_a[i][action];
            values_out[i] = values_a[i];
        }
    } else {
        torch::Tensor actions = logits.argmax(/*dim=*/1);
        torch::Tensor logprobs = log_softmax.gather(1, actions.unsqueeze(1)).squeeze(1);
        auto actions_a = actions.accessor<int64_t, 1>();
        auto logprobs_a = logprobs.accessor<float, 1>();
        auto values_a = values.accessor<float, 1>();
        for (int i = 0; i < batch_size; ++i) {
            actions_out[i] = actions_a[i];
            logprobs_out[i] = logprobs_a[i];
            values_out[i] = values_a[i];
        }
    }
}

}  // namespace park
