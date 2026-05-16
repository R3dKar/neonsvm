#include "neonsvm/decision_function/polynomial.hpp"
#include <Eigen/Dense>
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
    m_vectors_scaled = Eigen::MatrixXf::Zero(support_vectors.size(), support_vectors[0].size());

    for (size_t i = 0; i < support_vectors.size(); i++) {
      const Eigen::Map<const Eigen::RowVectorXf> support_vector(support_vectors[i].data(), support_vectors[i].size());
      m_vectors_scaled.row(i) = support_vector * gamma;
    }
  }

  float PolynomialDecisionFunction::PredictFromDot(Eigen::Ref<Eigen::VectorXf> dot) const {
    dot.array() += m_r;

    float32x4_t sum_v = {m_bias, 0, 0, 0};

    size_t i = 0;
    for (; i + 4 <= dot.size(); i += 4) {
      float32x4_t base_v = vld1q_f32(&dot[i]);
      float32x4_t pow_v = vdupq_n_f32(1);

      if (m_degree == 1) {
        pow_v = base_v;
      } else if (m_degree == 2) {
        pow_v = vmulq_f32(base_v, base_v);
      } else if (m_degree == 3) {
        pow_v = vmulq_f32(vmulq_f32(base_v, base_v), base_v);
      } else {
        uint32_t degree = m_degree;

        while (degree > 0) {
          if (degree % 2 == 1) pow_v = vmulq_f32(pow_v, base_v);

          base_v = vmulq_f32(base_v, base_v);
          degree >>= 1;
        }
      }

      const float32x4_t coeff_v = vld1q_f32(&m_coefficients[i]);
      sum_v = vfmaq_f32(sum_v, coeff_v, pow_v);
    }

    float result = vaddvq_f32(sum_v);

    for (; i < dot.size(); i++) {
      float base = dot[i];
      float pow = 1;

      if (m_degree == 1) {
        pow = base;
      } else if (m_degree == 2) {
        pow = base * base;
      } else if (m_degree == 3) {
        pow = base * base * base;
      } else {
        uint32_t degree = m_degree;

        while (degree > 0) {
          if (degree % 2 == 1) pow *= base;

          base *= base;
          degree >>= 1;
        }
      }

      result += pow * m_coefficients[i];
    }

    return result;
  }

  float PolynomialDecisionFunction::Predict(const std::vector<float>& features) const {
    const Eigen::Map<const Eigen::VectorXf> features_map(features.data(), features.size());
    Eigen::VectorXf dot = m_vectors_scaled * features_map;
    return PredictFromDot(dot);
  }

  std::vector<float> PolynomialDecisionFunction::PredictBatch(const std::vector<std::vector<float>>& batch) const {
    Eigen::MatrixXf matrix_batch(batch[0].size(), batch.size());
    for (size_t i = 0; i < batch.size(); i++) {
      matrix_batch.col(i) = Eigen::Map<const Eigen::VectorXf>(batch[i].data(), batch[i].size());
    }

    Eigen::MatrixXf dots = m_vectors_scaled * matrix_batch;

    std::vector<float> results(batch.size(), 0);

    for (size_t i = 0; i < batch.size(); i++) {
      results[i] = PredictFromDot(dots.col(i));
    }

    return results;
  }
} // namespace neonsvm
