#ifndef _NEONSVM_DECISION_FUNCTION_BASE_HPP_
#define _NEONSVM_DECISION_FUNCTION_BASE_HPP_

#include <vector>

namespace neonsvm {
  class DecisionFunctionBase {
  public:
    virtual float Predict(const std::vector<float>& features) const = 0;
    virtual std::vector<float> PredictBatch(const std::vector<std::vector<float>>& batch) const;

    virtual ~DecisionFunctionBase() = default;
  };
} // namespace neonsvm

#endif
