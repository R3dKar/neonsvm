#ifndef _NEONSVM_DECISION_FUNCTION_LINEAR_HPP_
#define _NEONSVM_DECISION_FUNCTION_LINEAR_HPP_

#include "neonsvm/decision_function/base.hpp"
#include <Eigen/Dense>

namespace neonsvm {
  /**
   * @brief Decision function with linear kernel (kernel="linear" from sklearn). K(x, v) = dot(x, v)
   */
  class LinearDecisionFunction : public DecisionFunctionBase {
  public:
    /**
     * @brief Construct a new `LinearDecisionFunction` object from dual coefficients, support vectors and bias.
     *
     * @param coefficients Dual coefficients of vectors.
     * @param support_vectors List of support vectors that corresponds to `coefficients`.
     * @param bias Bias for overall sum.
     */
    LinearDecisionFunction(const std::vector<float>& coefficients, const std::vector<std::vector<float>>& support_vectors, float bias);

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
    Eigen::VectorXf m_hyperplane;
    float m_bias;
  };
} // namespace neonsvm

#endif
