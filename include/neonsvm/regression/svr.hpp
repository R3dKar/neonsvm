#ifndef _NEONSVM_REGRESSION_SVR_HPP_
#define _NEONSVM_REGRESSION_SVR_HPP_

#include "neonsvm/decision_function/base.hpp"
#include <memory>
#include <vector>

namespace neonsvm {
  class SVR {
  public:
    SVR(std::unique_ptr<DecisionFunctionBase>&& decision_function);

    template <typename T, typename... Args>
    SVR(Args&&... args) : SVR(std::unique_ptr<T>(new T(std::forward<Args>(args)...))) {
    }

    float Predict(const std::vector<float>& features) const;
    std::vector<float> PredictBatch(const std::vector<std::vector<float>>& batch) const;

  private:
    std::unique_ptr<DecisionFunctionBase> m_decision_function;
  };
} // namespace neonsvm

#endif
