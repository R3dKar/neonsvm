#ifndef _NEONSVM_CLASSIFICATION_BASE_HPP_
#define _NEONSVM_CLASSIFICATION_BASE_HPP_

#include <cstdint>
#include <vector>

namespace neonsvm {
  /**
   * @brief Abstract class of classifier. Incapsulates logic that is responsible for labels assignment and probability.
   */
  class ClassifierBase {
  public:
    /**
     * @brief Calculates a class label from a single `features` vector.
     *
     * @param features Single vector of features.
     * @return Predicted class label.
     */
    virtual uint32_t PredictLabel(const std::vector<float>& features) const = 0;

    /**
     * @brief Calculates class labels from a batch of features vectors.
     *
     * @param batch List of features vectors.
     * @return Predicted class labels that corresponds to `batch` features vectors.
     */
    virtual std::vector<uint32_t> PredictLabelBatch(const std::vector<std::vector<float>>& batch) const;

    /**
     * @brief Calculates probabilites of a single `features` vector to be each class.
     *
     * @param features Single vector of features.
     * @return Predicted probabilities for each class. The layout is mostly this: for class 0 corresponding probability is at index 0, for class 1 at
     * index 1 etc.
     */
    virtual std::vector<float> PredictProbability(const std::vector<float>& features) const = 0;

    /**
     * @brief Calculates probabilities for each features vector of `batch` to be each class.
     *
     * @param batch List of features vectors.
     * @return List of predicted probabilities for each class that corresponds to `batch` features vectors. The layout of probabilities is mostly
     * this: for class 0 corresponding probability is at index 0, for class 1 at index 1 etc.
     */
    virtual std::vector<std::vector<float>> PredictProbabilityBatch(const std::vector<std::vector<float>>& batch) const;

    virtual ~ClassifierBase() = default;
  };
} // namespace neonsvm

#endif
