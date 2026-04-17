#include "neonsvm/decision_function/rbf.hpp"
#include "neonsvm/utility.hpp"
#include <arm_neon.h>
#include <cmath>
#include <cstddef>

namespace neonsvm {
  RBFDecisionFunction::RBFDecisionFunction(const std::vector<float>& coefficients, const std::vector<std::vector<float>>& support_vectors, float bias,
                                           float gamma) {
    m_bias = bias;
    m_sqrt_gamma = std::sqrt(gamma);
    m_coeffictients = coefficients;
    m_vectors_scaled = support_vectors;

    for (auto& vector : m_vectors_scaled) {
      vector.resize(neonsvm::utility::align_size(vector.size(), 4), 0);
      for (auto& item : vector) {
        item *= m_sqrt_gamma;
      }
    }
  }

  float RBFDecisionFunction::Predict(const std::vector<float>& features) const {
    std::vector<float> features_scaled(m_vectors_scaled.front().size(), 0);

    const float32x4_t sqrt_gamma_v = vdupq_n_f32(m_sqrt_gamma);

    size_t i = 0;
    for (; i + 4 <= features.size(); i += 4) {
      float32x4_t feature_v = vld1q_f32(&features[i]);
      feature_v = vmulq_f32(feature_v, sqrt_gamma_v);
      vst1q_f32(&features_scaled[i], feature_v);
    }
    for (; i < features.size(); i++) {
      features_scaled[i] = features[i] * m_sqrt_gamma;
    }

    std::vector<float> norms(m_vectors_scaled.size(), 0);
    for (size_t i = 0; i < norms.size(); i++) {
      const auto& support_vector = m_vectors_scaled[i];
      float32x4_t sum_v = vdupq_n_f32(0);
      for (size_t j = 0; j + 4 <= features_scaled.size(); j += 4) {
        const float32x4_t feature_v = vld1q_f32(&features_scaled[j]);
        const float32x4_t support_v = vld1q_f32(&support_vector[j]);
        const float32x4_t diff_v = vsubq_f32(feature_v, support_v);
        sum_v = vfmaq_f32(sum_v, diff_v, diff_v);
      }

      norms[i] = vaddvq_f32(sum_v);
    }

    float32x4_t sum_v = {m_bias, 0, 0, 0};

    i = 0;
    for (; i + 4 <= norms.size(); i += 4) {
      const float32x4_t norm_v = vld1q_f32(&norms[i]);
      const float32x4_t coeff_v = vld1q_f32(&m_coeffictients[i]);
      const float32x4_t exp_v = neonsvm::utility::vexpq_f32(vnegq_f32(norm_v));
      sum_v = vfmaq_f32(sum_v, exp_v, coeff_v);
    }

    float result = vaddvq_f32(sum_v);

    for (; i < norms.size(); i++) {
      result += m_coeffictients[i] * std::exp(-norms[i]);
    }

    return result;
  }
} // namespace neonsvm
