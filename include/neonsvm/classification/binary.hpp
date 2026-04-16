#ifndef _NEONSVM_CLASSIFICATION_BINARY_HPP_
#define _NEONSVM_CLASSIFICATION_BINARY_HPP_

#include "neonsvm/classification/base.hpp"
#include "neonsvm/decision_function/base.hpp"
#include <memory>
#include <tuple>

namespace neonsvm {
  class BinarySVC : public ClassifierBase {
  public:
    BinarySVC(std::unique_ptr<neonsvm::DecisionFunctionBase>&& decision_function);

    float PredictScore(const std::vector<float>& features) const;
    std::vector<float> PredictScoreBatch(const std::vector<std::vector<float>>& batch) const;

    virtual uint32_t PredictLabel(const std::vector<float>& features) const override;
    virtual std::vector<float> PredictProbability(const std::vector<float>& features) const override;

  private:
    std::unique_ptr<neonsvm::DecisionFunctionBase> m_decision_function;
  };

  class PlattBinarySVC : public BinarySVC {
  public:
    PlattBinarySVC(std::unique_ptr<neonsvm::DecisionFunctionBase>&& decision_function, float a, float b);

    virtual std::vector<float> PredictProbability(const std::vector<float>& features) const override;

  private:
    float m_a;
    float m_b;
  };
} // namespace neonsvm

#endif
