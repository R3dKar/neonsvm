#ifndef _NEONSVM_DECISION_FUNCTION_SIGMOID_HPP_
#define _NEONSVM_DECISION_FUNCTION_SIGMOID_HPP_

#include "neonsvm/decision_function/base.hpp"

namespace neonsvm {
  class SigmoidDecisionFunction : public DecisionFunctionBase {
  public:
    SigmoidDecisionFunction(const std::vector<float>& coefficients, const std::vector<std::vector<float>>& support_vectors, float bias, float gamma,
                            float r);

    float Predict(const std::vector<float>& features) const override;

  private:
    std::vector<float> m_coefficients;
    std::vector<std::vector<float>> m_vectors_scaled;
    float m_bias;
    float m_2r;
  };
} // namespace neonsvm

#endif
