#ifndef _NEONSVM_DECISION_FUNCTION_POLYNOMIAL_HPP_
#define _NEONSVM_DECISION_FUNCTION_POLYNOMIAL_HPP_

#include "neonsvm/decision_function/base.hpp"
#include <cstdint>

namespace neonsvm {
  class PolynomialDecisionFunction : public DecisionFunctionBase {
  public:
    PolynomialDecisionFunction(const std::vector<float>& coefficients, const std::vector<std::vector<float>>& support_vectors, float bias,
                               float gamma, float r, uint64_t degree);

    virtual float Predict(const std::vector<float>& features) const override;

  private:
    std::vector<float> m_coefficients;
    std::vector<std::vector<float>> m_vectors_scaled;
    float m_bias;
    float m_r;
    uint64_t m_degree;
  };
} // namespace neonsvm

#endif
