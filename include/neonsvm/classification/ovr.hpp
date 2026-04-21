#ifndef _NEONSVM_CLASSIFICATION_OVR_HPP_
#define _NEONSVM_CLASSIFICATION_OVR_HPP_

#include "neonsvm/classification/base.hpp"
#include "neonsvm/classification/binary.hpp"

#include <memory>
#include <vector>

namespace neonsvm {
  /**
   * @brief Multiclass classifier that uses "one vs rest" strategy. Uses `n` classifiers where `n` - number of classes.
   *
   * Note that probabilities don't sum up to 1 for this classifier but that will be fixed later. But since sklearn don't train "one vs rest"
   * classifiers with probabilities it shouldn't be a problem.
   */
  class OVRClassifier : public ClassifierBase {
  public:
    /**
     * @brief Construct a new `OVRClassifier` object from list of binary classifiers.
     *
     * The order of classifers matters. Consider case with 4 classes classifier. Thus there will be 4 binary classifiers with this layout:
     *
     * - 0 vs rest (positive for class 0)
     * 
     * - 1 vs rest (positive for class 1)
     * 
     * - 2 vs rest (positive for class 2)
     * 
     * - 3 vs rest (positive for class 3)
     *
     * @param classifiers List of binary classifier objects.
     */
    OVRClassifier(std::vector<std::unique_ptr<BinarySVC>>&& classifiers);

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
  };
} // namespace neonsvm

#endif
