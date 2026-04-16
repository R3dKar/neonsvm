#ifndef _NEONSVM_CLASSIFICATION_BASE_HPP_
#define _NEONSVM_CLASSIFICATION_BASE_HPP_

#include <cstdint>
#include <vector>

namespace neonsvm {
  class ClassifierBase {
  public:
    virtual uint32_t PredictLabel(const std::vector<float>& features) const = 0;
    virtual std::vector<uint32_t> PredictLabelBatch(const std::vector<std::vector<float>>& batch) const;

    virtual std::vector<float> PredictProbability(const std::vector<float>& features) const = 0;
    virtual std::vector<std::vector<float>> PredictProbabilityBatch(const std::vector<std::vector<float>>& batch) const;

    virtual ~ClassifierBase() = default;
  };
} // namespace neonsvm

#endif
