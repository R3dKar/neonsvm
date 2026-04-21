#ifndef _NEONSVM_DECISION_FUNCTION_BASE_HPP_
#define _NEONSVM_DECISION_FUNCTION_BASE_HPP_

#include <vector>

namespace neonsvm {
  /**
   * @brief Abstract class of a decision function. Incapsulates most of logic used by SVM models.
   */
  class DecisionFunctionBase {
  public:
    /**
     * @brief Calculates value of decision function (aka score) from a single `features` vectors.
     *
     * @param features Single vector of features.
     * @return Predicted decision function value.
     */
    virtual float Predict(const std::vector<float>& features) const = 0;

    /**
     * @brief Calculates values of decision function (aka score) from a `batch` of features vectors.
     *
     * @param batch List of features vectors.
     * @return Predicted decision function values that corresponds to `batch` features vectors.
     */
    virtual std::vector<float> PredictBatch(const std::vector<std::vector<float>>& batch) const;

    virtual ~DecisionFunctionBase() = default;
  };
} // namespace neonsvm

#endif
