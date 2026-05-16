#ifndef _NEONSVM_DECISION_FUNCTION_POLYNOMIAL_HPP_
#define _NEONSVM_DECISION_FUNCTION_POLYNOMIAL_HPP_

#include "neonsvm/decision_function/base.hpp"
#include <Eigen/Dense>
#include <cstdint>

namespace neonsvm {
  /**
   * @brief Decision function with polynimial kernel (kernel="poly" from sklearn). K(x, v) = (g*dot(x, v) + r)^d
   */
  class PolynomialDecisionFunction : public DecisionFunctionBase {
  public:
    /**
     * @brief Construct a new `PolynomialDecisionFunction` object from coefficients, support vectors, bias and kernel parameters: gamma, r (bias) and
     * degree
     *
     * @param coefficients Dual coefficients of vectors.
     * @param support_vectors List of support vectors that corresponds to `coefficients`.
     * @param bias Bias for overall sum.
     * @param gamma Kernel parameter gamma.
     * @param r Kernel parameter r (bias).
     * @param degree Kernel parameter degree.
     */
    PolynomialDecisionFunction(const std::vector<float>& coefficients, const std::vector<std::vector<float>>& support_vectors, float bias,
                               float gamma, float r, uint32_t degree);

    /**
     * @brief Calculates value of decision function (aka score) from a single `features` vectors.
     *
     * @param features Single vector of features.
     * @return Predicted decision function value.
     */
    float Predict(const std::vector<float>& features) const override;

    /**
     * @brief Calculates values of decision function (aka score) from a `batch` of features vectors.
     *
     * @param batch List of features vectors.
     * @return Predicted decision function values that corresponds to `batch` features vectors.
     */
    std::vector<float> PredictBatch(const std::vector<std::vector<float>>& batch) const override;

  private:
    float PredictFromDot(Eigen::Ref<Eigen::VectorXf> dot) const;

    std::vector<float> m_coefficients;
    Eigen::MatrixXf m_vectors_scaled;
    float m_bias;
    float m_r;
    uint32_t m_degree;
  };
} // namespace neonsvm

#endif
