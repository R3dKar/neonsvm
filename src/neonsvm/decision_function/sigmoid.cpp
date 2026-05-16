#include "neonsvm/decision_function/sigmoid.hpp"
#include "neonsvm/utility.hpp"
#include <Eigen/Dense>
#include <arm_neon.h>
#include <cmath>
#include <cstddef>

namespace neonsvm {
  SigmoidDecisionFunction::SigmoidDecisionFunction(const std::vector<float>& coefficients, const std::vector<std::vector<float>>& support_vectors,
                                                   float bias, float gamma, float r) {
    m_bias = bias;
    m_2r = 2 * r;
    m_coefficients = coefficients;
    m_vectors_scaled = Eigen::MatrixXf::Zero(support_vectors.size(), support_vectors[0].size());

    for (size_t i = 0; i < support_vectors.size(); i++) {
      const Eigen::Map<const Eigen::RowVectorXf> support_vector(support_vectors[i].data(), support_vectors[i].size());
      m_vectors_scaled.row(i) = support_vector * 2 * gamma;
    }
  }

  float SigmoidDecisionFunction::PredictFromDot(Eigen::Ref<Eigen::VectorXf> dot) const {
    dot.array() += m_2r;

    float32x4_t sum_v = {m_bias, 0, 0, 0};
    const float32x4_t one_v = vdupq_n_f32(1);

    size_t i = 0;
    for (; i + 4 <= dot.size(); i += 4) {
      const float32x4_t coeff_v = vld1q_f32(&m_coefficients[i]);
      const float32x4_t x_v = vld1q_f32(&dot[i]);
      const float32x4_t exp_v = utility::vexpq_f32(x_v);
      const float32x4_t tanh_v = vdivq_f32(vsubq_f32(exp_v, one_v), vaddq_f32(exp_v, one_v));
      sum_v = vfmaq_f32(sum_v, coeff_v, tanh_v);
    }

    float result = vaddvq_f32(sum_v);

    for (; i < dot.size(); i++) {
      result += m_coefficients[i] * std::tanh(dot[i] / 2.0);
    }

    return result;
  }

  float SigmoidDecisionFunction::Predict(const std::vector<float>& features) const {
    const Eigen::Map<const Eigen::VectorXf> features_map(features.data(), features.size());
    Eigen::VectorXf dot = m_vectors_scaled * features_map;
    return PredictFromDot(dot);
  }

  std::vector<float> SigmoidDecisionFunction::PredictBatch(const std::vector<std::vector<float>>& batch) const {
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
