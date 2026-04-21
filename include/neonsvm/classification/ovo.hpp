#ifndef _NEONSVM_CLASSIFICATION_OVO_HPP_
#define _NEONSVM_CLASSIFICATION_OVO_HPP_

#include "neonsvm/classification/base.hpp"
#include "neonsvm/classification/binary.hpp"

#include <memory>
#include <vector>

namespace neonsvm {
  /**
   * @brief Multiclass classifier that uses "one vs one" strategy. Uses `n*(n - 1)/2` classifiers where `n` - number of classes.
   */
  class OVOClassifier : public ClassifierBase {
  public:
    /**
     * @brief Construct a new `OVOClassifier` object from list of binary classifiers.
     *
     * The order of classifiers matters. Consider case with 4 classes classifier. Thus there will be 6 binary classifiers with this layout:
     *
     * - 0 vs 1 (positive for class 0)
     *
     * - 0 vs 2 (positive for class 0)
     *
     * - 0 vs 3 (positive for class 0)
     *
     * - 1 vs 2 (positive for class 1)
     *
     * - 1 vs 3 (positive for class 1)
     *
     * - 2 vs 3 (positive for class 2)
     *
     * @param classifiers List of binary classifier objects.
     */
    OVOClassifier(std::vector<std::unique_ptr<BinarySVC>>&& classifiers);

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
    std::vector<std::unique_ptr<BinarySVC>> m_classifiers;
    size_t m_class_count;
  };
} // namespace neonsvm

#endif
