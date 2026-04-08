#include "neonsvm/decision_function/linear.hpp"
#include <arm_neon.h>
#include <cstddef>

namespace neonsvm {
  LinearDecisionFunction::LinearDecisionFunction(const std::vector<float>& coefficients, const std::vector<std::vector<float>>& support_vectors,
                                                 float bias) {
    m_bias = bias;

    m_hyperplane = std::vector<float>(coefficients.size(), 0);
    for (size_t i = 0; i < coefficients.size(); i++)
      for (size_t j = 0; j < support_vectors[i].size(); j++) m_hyperplane[j] += coefficients[i] * support_vectors[i][j];
  }

  float LinearDecisionFunction::Predict(const std::vector<float>& features) const {
    float32x4_t sum_v = {m_bias, 0, 0, 0};

    size_t i = 0;
    for (; i + 4 <= m_hyperplane.size(); i += 4) {
      const float32x4_t feature_v = vld1q_f32(&features[i]);
      const float32x4_t hyperplane_v = vld1q_f32(&m_hyperplane[i]);
      sum_v = vfmaq_f32(sum_v, feature_v, hyperplane_v);
    }

    float result = vaddvq_f32(sum_v);

    for (; i < m_hyperplane.size(); i++) result += features[i] * m_hyperplane[i];

    return result;
  }
} // namespace neonsvm
