#include "neonsvm/decision_function/sigmoid.hpp"
#include "neonsvm/utility.hpp"
#include <arm_neon.h>
#include <cmath>
#include <cstddef>

namespace neonsvm {
  SigmoidDecisionFunction::SigmoidDecisionFunction(const std::vector<float>& coefficients, const std::vector<std::vector<float>>& support_vectors,
                                                   float bias, float gamma, float r) {
    m_bias = bias;
    m_2r = 2 * r;
    m_coefficients = coefficients;
    m_vectors_scaled = support_vectors;

    for (auto& vector : m_vectors_scaled) {
      for (auto& item : vector) {
        item *= 2 * gamma;
      }
    }
  }

  float SigmoidDecisionFunction::Predict(const std::vector<float>& features) const {
    std::vector<float> xs(m_vectors_scaled.size(), 0);
    for (size_t i = 0; i < xs.size(); i++) {
      const auto& support_vector = m_vectors_scaled[i];
      float32x4_t x_v = {m_2r, 0, 0, 0};

      size_t j = 0;
      for (; j + 4 <= features.size(); j += 4) {
        const float32x4_t feature_v = vld1q_f32(&features[j]);
        const float32x4_t support_v = vld1q_f32(&support_vector[j]);
        x_v = vfmaq_f32(x_v, feature_v, support_v);
      }

      xs[i] = vaddvq_f32(x_v);
      for (; j < features.size(); j++) {
        xs[i] += features[j] * support_vector[j];
      }
    }

    float32x4_t sum_v = {m_bias, 0, 0, 0};

    size_t i = 0;
    for (; i + 4 <= xs.size(); i += 4) {
      const float32x4_t coeff_v = vld1q_f32(&m_coefficients[i]);
      const float32x4_t x_v = vld1q_f32(&xs[i]);
      const float32x4_t exp_v = utility::vexpq_f32(x_v);
      const float32x4_t one_v = vdupq_n_f32(1);
      const float32x4_t tanh_v = vdivq_f32(vsubq_f32(exp_v, one_v), vaddq_f32(exp_v, one_v));
      sum_v = vfmaq_f32(sum_v, coeff_v, tanh_v);
    }

    float result = vaddvq_f32(sum_v);

    for (; i < xs.size(); i++) {
      result += m_coefficients[i] * std::tanh(xs[i] / 2.0);
    }

    return result;
  }
} // namespace neonsvm
