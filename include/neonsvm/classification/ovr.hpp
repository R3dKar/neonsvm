#ifndef _NEONSVM_CLASSIFICATION_OVR_HPP_
#define _NEONSVM_CLASSIFICATION_OVR_HPP_

#include "neonsvm/classification/base.hpp"
#include "neonsvm/classification/binary.hpp"

#include <memory>
#include <vector>

namespace neonsvm {
  class OVRClassifier : public ClassifierBase {
  public:
    OVRClassifier(std::vector<std::unique_ptr<BinarySVC>>&& classifiers);

    virtual uint32_t PredictLabel(const std::vector<float>& features) const override;
    virtual std::vector<float> PredictProbability(const std::vector<float>& features) const override;

  private:
    std::vector<std::unique_ptr<BinarySVC>> m_classifiers;
  };
} // namespace neonsvm

#endif
