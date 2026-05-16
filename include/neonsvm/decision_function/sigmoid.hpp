#ifndef _NEONSVM_DECISION_FUNCTION_SIGMOID_HPP_
#define _NEONSVM_DECISION_FUNCTION_SIGMOID_HPP_

#include "neonsvm/decision_function/base.hpp"
#include <Eigen/Dense>

namespace neonsvm {
  /**
   * @brief Decision function with sigmoid kernel (kernel="sigmoid" from sklearn). K(x, v) = tanh(g*dot(x, v) + r)
   */
  class SigmoidDecisionFunction : public DecisionFunctionBase {
  public:
    /**
     * @brief Construct a new `SigmoidDecisionFunction` object from coefficients, support vectors, bias and kernel parameters gamma and r (bias).
     *
     * @param coefficients Dual coefficients of vectors.
     * @param support_vectors List of support vectors that corresponds to `coefficients`.
     * @param bias Bias for overall sum.
     * @param gamma Kernel parameter gamma.
     * @param r Kernel parameter r (bias).
     */
    SigmoidDecisionFunction(const std::vector<float>& coefficients, const std::vector<std::vector<float>>& support_vectors, float bias, float gamma,
                            float r);

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
    float m_2r;
  };
} // namespace neonsvm

#endif
