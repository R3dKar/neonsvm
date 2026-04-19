#include "neonsvm/classification/ovo.hpp"
#include "neonsvm/utility.hpp"
#include <cmath>

namespace neonsvm {
  OVOClassifier::OVOClassifier(std::vector<std::unique_ptr<BinarySVC>>&& classifiers) : m_classifiers(std::move(classifiers)) {
    m_class_count = static_cast<size_t>(std::round((std::sqrt(8 * m_classifiers.size() + 1) + 1) / 2));

    // TODO: throw exception if m_classifiers size is wrong
  }

  uint32_t OVOClassifier::PredictLabel(const std::vector<float>& features) const {
    std::vector<size_t> votes(m_class_count, 0);

    size_t classifier_index = 0;
    for (size_t i = 0; i < m_class_count - 1; i++) {
      for (size_t j = i + 1; j < m_class_count; j++) {
        const auto ovo_label = m_classifiers[classifier_index]->PredictLabel(features);

        if (ovo_label == 1)
          votes[i]++;
        else
          votes[j]++;

        classifier_index++;
      }
    }

    uint32_t label = 0;

    for (size_t i = 0; i < votes.size(); i++) {
      if (votes[i] > votes[label]) label = i;
    }

    return label;
  }

  std::vector<float> OVOClassifier::PredictProbability(const std::vector<float>& features) const {
    std::vector<std::vector<float>> ovo_probabilities{};
    ovo_probabilities.reserve(m_class_count);

    for (size_t i = 0; i < m_class_count; i++) {
      ovo_probabilities.emplace_back(std::vector<float>(m_class_count, 0));
    }

    size_t classifier_index = 0;
    for (size_t i = 0; i < m_class_count - 1; i++) {
      for (size_t j = i + 1; j < m_class_count; j++) {
        constexpr float min_probability = 1e-7f;
        constexpr float max_probability = 1 - min_probability;
        const auto vs_probabilities = m_classifiers[classifier_index]->PredictProbability(features);

        ovo_probabilities[j][i] = utility::clamp(vs_probabilities[0], min_probability, max_probability);
        ovo_probabilities[i][j] = utility::clamp(vs_probabilities[1], min_probability, max_probability);

        classifier_index++;
      }
    }

    // https://github.com/scikit-learn/scikit-learn/blob/20c0129906b26803f810e41936e5487d5becfcf4/sklearn/svm/src/libsvm/svm.cpp#L2043
    std::vector<float> probabilities(m_class_count, 1.0f / m_class_count);

    std::vector<std::vector<float>> Q{};
    Q.reserve(m_class_count);
    for (size_t i = 0; i < m_class_count; i++) {
      Q.emplace_back(std::vector<float>(m_class_count, 0));
    }

    std::vector<float> Qp(m_class_count, 0);
    float pQp = 0;
    const float eps = 0.005f / m_class_count;
    const size_t max_iteration = std::max<size_t>(100, m_class_count);

    for (size_t i = 0; i < m_class_count; i++) {
      for (size_t j = 0; j < i; j++) {
        Q[i][i] += ovo_probabilities[j][i] * ovo_probabilities[j][i];
        Q[i][j] = Q[j][i];
      }

      for (size_t j = i + 1; j < m_class_count; j++) {
        Q[i][i] += ovo_probabilities[j][i] * ovo_probabilities[j][i];
        Q[i][j] = -ovo_probabilities[j][i] * ovo_probabilities[i][j];
      }
    }

    for (size_t iteration = 0; iteration < max_iteration; iteration++) {
      pQp = 0;

      for (size_t i = 0; i < m_class_count; i++) {
        Qp[i] = 0;
        for (size_t j = 0; j < m_class_count; j++) {
          Qp[i] += Q[i][j] * probabilities[j];
        }
        pQp += probabilities[i] * Qp[i];
      }

      float max_error = 0;
      for (size_t i = 0; i < m_class_count; i++) {
        const float error = std::abs(Qp[i] - pQp);
        max_error = std::max(max_error, error);
      }
      if (max_error < eps) break;

      for (size_t i = 0; i < m_class_count; i++) {
        const float diff = (-Qp[i] + pQp) / Q[i][i];
        probabilities[i] += diff;
        pQp = (pQp + diff * (diff * Q[i][i] + 2 * Qp[i])) / (1 + diff) / (1 + diff);

        for (size_t j = 0; j < m_class_count; j++) {
          Qp[j] = (Qp[j] + diff * Q[i][j]) / (1 + diff);
          probabilities[j] /= 1 + diff;
        }
      }
    }

    return probabilities;
  }
} // namespace neonsvm
