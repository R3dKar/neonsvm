#include "neonsvm/decision_function/base.hpp"
#include <cstddef>

namespace neonsvm {
  std::vector<float> DecisionFunctionBase::PredictBatch(const std::vector<std::vector<float>>& batch) const {
    std::vector<float> results(batch.size(), 0);

    for (size_t i = 0; i < batch.size(); i++) results[i] = Predict(batch[i]);

    return results;
  }
} // namespace neonsvm
