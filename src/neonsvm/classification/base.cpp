#include "neonsvm/classification/base.hpp"
#include <cstddef>

namespace neonsvm {
  std::vector<uint32_t> ClassifierBase::PredictLabelBatch(const std::vector<std::vector<float>>& batch) const {
    std::vector<uint32_t> results(batch.size(), 0);

    for (size_t i = 0; i < batch.size(); i++) results[i] = PredictLabel(batch[i]);

    return results;
  }

  std::vector<std::vector<float>> ClassifierBase::PredictProbabilityBatch(const std::vector<std::vector<float>>& batch) const {
    std::vector<std::vector<float>> results{};
    results.reserve(batch.size());

    for (size_t i = 0; i < batch.size(); i++) results.emplace_back(PredictProbability(batch[i]));

    return results;
  }
} // namespace neonsvm
