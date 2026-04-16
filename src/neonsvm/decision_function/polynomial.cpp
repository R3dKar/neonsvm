#include "neonsvm/decision_function/polynomial.hpp"
#include <arm_neon.h>
#include <cstddef>

namespace neonsvm {
  PolynomialDecisionFunction::PolynomialDecisionFunction(const std::vector<float>& coefficients,
                                                         const std::vector<std::vector<float>>& support_vectors, float bias, float gamma, float r,
                                                         uint32_t degree) {
    m_bias = bias;
    m_r = r;
    m_degree = degree;
    m_coefficients = coefficients;
    m_vectors_scaled = support_vectors;

    for (auto& vector : m_vectors_scaled)
      for (auto& item : vector) item *= gamma;
  }

  float PolynomialDecisionFunction::Predict(const std::vector<float>& features) const {
    std::vector<float> bases(features.size(), 0);
    for (size_t i = 0; i < features.size(); i++) {
      const auto& support_vector = m_vectors_scaled[i];
      float32x4_t sum_v = {m_r, 0, 0, 0};

      size_t j = 0;
      for (; j + 4 <= support_vector.size(); j += 4) {
        const float32x4_t support_v = vld1q_f32(&support_vector[j]);
        const float32x4_t feature_v = vld1q_f32(&features[j]);
        sum_v = vfmaq_f32(sum_v, support_v, feature_v);
      }

      bases[i] = vaddvq_f32(sum_v);
      for (; j < support_vector.size(); j++) bases[i] += support_vector[j] * features[j];
    }

    float32x4_t sum_v = {m_bias, 0, 0, 0};

    size_t i = 0;
    for (; i + 4 <= bases.size(); i += 4) {
      float32x4_t base_v = vld1q_f32(&bases[i]);
      float32x4_t pow_v = vdupq_n_f32(1);
      uint32_t degree = m_degree;

      while (degree > 0) {
        if (degree % 2 == 1) pow_v = vmulq_f32(pow_v, base_v);

        base_v = vmulq_f32(base_v, base_v);
        degree >>= 1;
      }

      const float32x4_t coeff_v = vld1q_f32(&m_coefficients[i]);
      sum_v = vfmaq_f32(sum_v, coeff_v, pow_v);
    }

    float result = vaddvq_f32(sum_v);

    for (; i < bases.size(); i++) {
      float base = bases[i];
      float pow = 1;
      uint32_t degree = m_degree;

      while (degree > 0) {
        if (degree % 2 == 1) pow *= base;

        base *= base;
        degree >>= 1;
      }

      result += pow * m_coefficients[i];
    }

    return result;
  }
} // namespace neonsvm
