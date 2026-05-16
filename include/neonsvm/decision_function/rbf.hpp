#ifndef _NEONSVM_DECISION_FUNCTION_RBF_HPP_
#define _NEONSVM_DECISION_FUNCTION_RBF_HPP_

#include "neonsvm/decision_function/base.hpp"

namespace neonsvm {
  /**
   * @brief Decision function with radial basis function kernel (kernel="rbf" from sklearn). K(x, v) = exp(-g*abs(x - v)^2)
   */
  class RBFDecisionFunction : public DecisionFunctionBase {
  public:
    /**
     * @brief Construct a new `RBFDecisionFunction` object from coefficients, support vectors, bias and kernel parameter gamma.
     *
     * @param coefficients Dual coefficients of vectors.
     * @param support_vectors List of support vectors that corresponds to `coefficients`.
     * @param bias Bias for overall sum.
     * @param gamma Kernel parameter gamma.
     */
    RBFDecisionFunction(const std::vector<float>& coefficients, const std::vector<std::vector<float>>& support_vectors, float bias, float gamma);

    /**
     * @brief Calculates value of decision function (aka score) from a single `features` vectors.
     *
     * @param features Single vector of features.
     * @return Predicted decision function value.
     */
    float Predict(const std::vector<float>& features) const override;

  private:
    std::vector<float> m_coeffictients;
    std::vector<std::vector<float>> m_vectors_scaled;
    float m_sqrt_gamma;
    float m_bias;
  };
} // namespace neonsvm

#endif
