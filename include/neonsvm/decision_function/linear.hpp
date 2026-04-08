#ifndef _NEONSVM_DECISION_FUNCTION_LINEAR_HPP_
#define _NEONSVM_DECISION_FUNCTION_LINEAR_HPP_

#include "neonsvm/decision_function/base.hpp"

namespace neonsvm {
  class LinearDecisionFunction : public DecisionFunctionBase {
  public:
    LinearDecisionFunction(const std::vector<float>& coefficients, const std::vector<std::vector<float>>& support_vectors, float bias);

    virtual float Predict(const std::vector<float>& features) const override;

  private:
    std::vector<float> m_hyperplane;
    float m_bias;
  };
} // namespace neonsvm

#endif
