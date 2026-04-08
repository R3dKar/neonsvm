#include "neonsvm/svr.hpp"

namespace neonsvm {
  SVR::SVR(std::unique_ptr<DecisionFunctionBase>&& decision_function) : m_decision_function(std::move(decision_function)) {
  }

  float SVR::Predict(const std::vector<float>& features) const {
    return m_decision_function->Predict(features);
  }

  std::vector<float> SVR::PredictBatch(const std::vector<std::vector<float>>& batch) const {
    return m_decision_function->PredictBatch(batch);
  }
} // namespace neonsvm
