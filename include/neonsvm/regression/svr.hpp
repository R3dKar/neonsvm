#ifndef _NEONSVM_REGRESSION_SVR_HPP_
#define _NEONSVM_REGRESSION_SVR_HPP_

#include "neonsvm/decision_function/base.hpp"
#include <memory>
#include <vector>

namespace neonsvm {
  /**
   * @brief Support Vector Regression. Simply wraps `DecisionFunctionBase` object and serve semantic purpose only.
   */
  class SVR {
  public:
    /**
     * @brief Construct a new `SVR` object from `DecisionFunctionBase` object.
     *
     * @param decision_function Decision function object pointer that holds the entire logic.
     */
    SVR(std::unique_ptr<DecisionFunctionBase>&& decision_function);

    /**
     * @brief Calculates value from a single `features` vector.
     *
     * @param features Single vector of features.
     * @return Predicted regression value.
     */
    float Predict(const std::vector<float>& features) const;

    /**
     * @brief Calcualtes values from a batch of features vectors.
     *
     * @param batch List of features vectors.
     * @return Predicted regression values that corresponds to `batch` features vectors.
     */
    std::vector<float> PredictBatch(const std::vector<std::vector<float>>& batch) const;

  private:
    std::unique_ptr<DecisionFunctionBase> m_decision_function;
  };
} // namespace neonsvm

#endif
