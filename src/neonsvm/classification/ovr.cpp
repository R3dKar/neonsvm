#include "neonsvm/classification/ovr.hpp"
#include <limits>
#include <vector>

namespace neonsvm {
  OVRClassifier::OVRClassifier(std::vector<std::unique_ptr<BinarySVC>>&& classifiers) : m_classifiers(std::move(classifiers)) {
  }

  uint32_t OVRClassifier::PredictLabel(const std::vector<float>& features) const {
    float max_score = std::numeric_limits<float>::lowest();
    uint32_t label = 0;

    for (size_t i = 0; i < m_classifiers.size(); i++) {
      const float score = m_classifiers[i]->PredictScore(features);

      if (max_score < score) {
        max_score = score;
        label = i;
      }
    }

    return label;
  }

  std::vector<float> OVRClassifier::PredictProbability(const std::vector<float>& features) const {
    std::vector<float> probabilities(m_classifiers.size(), 0);

    for (size_t i = 0; i < m_classifiers.size(); i++) {
      probabilities[i] = m_classifiers[i]->PredictProbability(features)[1];
    }

    return probabilities;
  }
} // namespace neonsvm
