#include "neonsvm/classification/binary.hpp"
#include <cmath>

namespace neonsvm {
  BinarySVC::BinarySVC(std::unique_ptr<neonsvm::DecisionFunctionBase>&& decision_function) : m_decision_function(std::move(decision_function)) {
  }

  float BinarySVC::PredictScore(const std::vector<float>& features) const {
    return m_decision_function->Predict(features);
  }

  std::vector<float> BinarySVC::PredictScoreBatch(const std::vector<std::vector<float>>& batch) const {
    return m_decision_function->PredictBatch(batch);
  }

  uint32_t BinarySVC::PredictLabel(const std::vector<float>& features) const {
    const float score = PredictScore(features);
    return (score > 0) ? 1 : 0;
  }

  std::vector<float> BinarySVC::PredictProbability(const std::vector<float>& features) const {
    const float score = PredictScore(features);

    if (score > 0)
      return {0, 1};
    else
      return {1, 0};
  }

  PlattBinarySVC::PlattBinarySVC(std::unique_ptr<neonsvm::DecisionFunctionBase>&& decision_function, float a, float b)
      : BinarySVC(std::move(decision_function)), m_a(a), m_b(b) {
  }

  std::vector<float> PlattBinarySVC::PredictProbability(const std::vector<float>& features) const {
    const float score = PredictScore(features);
    const float probability = 1.0f / (1.0f + std::exp(m_a * score + m_b));
    return {1 - probability, probability};
  }
} // namespace neonsvm
