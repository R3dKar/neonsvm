#ifndef _NEONSVM_DECISION_FUNCTION_RBF_HPP_
#define _NEONSVM_DECISION_FUNCTION_RBF_HPP_

#include "neonsvm/decision_function/base.hpp"

namespace neonsvm {
  class RBFDecisionFunction : public DecisionFunctionBase {
  public:
    RBFDecisionFunction(const std::vector<float>& coefficients, const std::vector<std::vector<float>>& support_vectors, float bias, float gamma);

    virtual float Predict(const std::vector<float>& features) const override;

  private:
    std::vector<float> m_coeffictients;
    std::vector<std::vector<float>> m_vectors_scaled;
    float m_sqrt_gamma;
    float m_bias;
  };
} // namespace neonsvm

#endif
