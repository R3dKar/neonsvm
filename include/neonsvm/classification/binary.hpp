#ifndef _NEONSVM_CLASSIFICATION_BINARY_HPP_
#define _NEONSVM_CLASSIFICATION_BINARY_HPP_

#include "neonsvm/classification/base.hpp"
#include "neonsvm/decision_function/base.hpp"
#include <memory>
#include <tuple>

namespace neonsvm {
  /**
   * @brief Binary support vector classifier. Simply wraps decision function object. This is a building block for a multiclass classifiers.
   */
  class BinarySVC : public ClassifierBase {
  public:
    /**
     * @brief Construct a new `BinarySVC` object from `DecisionFunctionBase` object.
     *
     * @param decision_function Decision function object pointer.
     */
    BinarySVC(std::unique_ptr<neonsvm::DecisionFunctionBase>&& decision_function);

    /**
     * @brief Calculates the value of decision function (aka score) from a single `features` vectors.
     *
     * @param features Single vector of features.
     * @return Predicted decision function value.
     */
    float PredictScore(const std::vector<float>& features) const;

    /**
     * @brief Calculates values of decision function (aka score) from a `batch` of features vectors.
     *
     * @param batch List of features vectors.
     * @return Predicted decision function values that corresponds to `batch` features vectors.
     */
    std::vector<float> PredictScoreBatch(const std::vector<std::vector<float>>& batch) const;

    /**
     * @brief Calculates a class label from a single `features` vector.
     *
     * @param features Single vector of features.
     * @return Predicted class label.
     */
    virtual uint32_t PredictLabel(const std::vector<float>& features) const override;

    /**
     * @brief Calculates probabilites of a single `features` vector to be each class.
     *
     * @param features Single vector of features.
     * @return Predicted probabilities for each class. The layout is mostly this: for class 0 corresponding probability is at index 0, for class 1 at
     * index 1 etc.
     */
    virtual std::vector<float> PredictProbability(const std::vector<float>& features) const override;

  private:
    std::unique_ptr<neonsvm::DecisionFunctionBase> m_decision_function;
  };

  /**
   * @brief `BinarySVC` variant with proper Platt probability handling.
   */
  class PlattBinarySVC : public BinarySVC {
  public:
    /**
     * @brief Construct a new `PlattBinarySVC` object from a decision function object and coefficients `a` and `b`.
     *
     * @param decision_function - Decision function object pointer.
     * @param a - Score scale coefficient.
     * @param b - Score bias coefficient.
     */
    PlattBinarySVC(std::unique_ptr<neonsvm::DecisionFunctionBase>&& decision_function, float a, float b);

    /**
     * @brief Calculates probabilites of a single `features` vector to be each class.
     *
     * @param features Single vector of features.
     * @return Predicted probabilities for each class. The layout is mostly this: for class 0 corresponding probability is at index 0, for class 1 at
     * index 1 etc.
     */
    virtual std::vector<float> PredictProbability(const std::vector<float>& features) const override;

  private:
    float m_a;
    float m_b;
  };
} // namespace neonsvm

#endif
