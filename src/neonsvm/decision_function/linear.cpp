#include "neonsvm/decision_function/linear.hpp"
#include <Eigen/Dense>
#include <arm_neon.h>
#include <cstddef>

namespace neonsvm {
  LinearDecisionFunction::LinearDecisionFunction(const std::vector<float>& coefficients, const std::vector<std::vector<float>>& support_vectors,
                                                 float bias) {
    m_bias = bias;

    m_hyperplane = Eigen::VectorXf::Zero(support_vectors[0].size());
    for (size_t i = 0; i < coefficients.size(); i++) {
      const Eigen::Map<const Eigen::VectorXf> support_vector(support_vectors[i].data(), support_vectors[i].size());
      m_hyperplane += coefficients[i] * support_vector;
    }
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

    for (; i < m_hyperplane.size(); i++) {
      result += features[i] * m_hyperplane[i];
    }

    return result;
  }

  std::vector<float> LinearDecisionFunction::PredictBatch(const std::vector<std::vector<float>>& batch) const {
    Eigen::MatrixXf matrix_batch(batch.size(), batch[0].size());
    for (size_t i = 0; i < batch.size(); i++) {
      matrix_batch.row(i) = Eigen::Map<const Eigen::RowVectorXf>(batch[i].data(), batch[i].size());
    }

    std::vector<float> results(batch.size(), 0);
    Eigen::Map<Eigen::VectorXf> results_map(results.data(), results.size());

    results_map = (matrix_batch * m_hyperplane).array() + m_bias;

    return results;
  }
} // namespace neonsvm
