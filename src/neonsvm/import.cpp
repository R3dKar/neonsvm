#include "neonsvm/import.hpp"
#include "neonsvm/classification/base.hpp"
#include "neonsvm/classification/binary.hpp"
#include "neonsvm/classification/ovo.hpp"
#include "neonsvm/classification/ovr.hpp"
#include "neonsvm/decision_function/base.hpp"
#include "neonsvm/decision_function/linear.hpp"
#include "neonsvm/decision_function/polynomial.hpp"
#include "neonsvm/decision_function/rbf.hpp"
#include "neonsvm/decision_function/sigmoid.hpp"
#include "neonsvm/regression/svr.hpp"
#include "neonsvm/utility.hpp"
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <yaml-cpp/yaml.h>

namespace {
  using namespace neonsvm;

  std::unique_ptr<DecisionFunctionBase> ParseDecisionFunction(const YAML::Node& data, const YAML::Node& kernel, bool inverse = false) {
    const float k = (inverse) ? -1.0f : 1.0f;

    const auto bias = k * data["bias"].as<float>();

    std::vector<std::vector<float>> support_vectors{};
    support_vectors.reserve(data["support-vectors"].size());

    std::vector<float> coefficients{};
    coefficients.reserve(data["support-vectors"].size());

    for (const auto vector_node : data["support-vectors"]) {
      const auto support_vector = vector_node["vector"].as<std::vector<float>>();
      const auto coefficient = k * vector_node["coefficient"].as<float>(1.0f);

      support_vectors.emplace_back(std::move(support_vector));
      coefficients.push_back(coefficient);
    }

    std::unique_ptr<DecisionFunctionBase> decision_function = nullptr;

    const auto kernel_type = kernel["type"].as<std::string>();
    if (kernel_type == "linear") {
      decision_function = std::unique_ptr<DecisionFunctionBase>(new LinearDecisionFunction(coefficients, support_vectors, bias));

    } else if (kernel_type == "polynomial") {
      const auto gamma = kernel["gamma"].as<float>();
      const auto r = kernel["bias"].as<float>();
      const auto degree = kernel["degree"].as<uint32_t>();
      decision_function =
          std::unique_ptr<DecisionFunctionBase>(new PolynomialDecisionFunction(coefficients, support_vectors, bias, gamma, r, degree));

    } else if (kernel_type == "rbf") {
      const auto gamma = kernel["gamma"].as<float>();
      decision_function = std::unique_ptr<DecisionFunctionBase>(new RBFDecisionFunction(coefficients, support_vectors, bias, gamma));

    } else if (kernel_type == "sigmoid") {
      const auto gamma = kernel["gamma"].as<float>();
      const auto r = kernel["bias"].as<float>();
      decision_function = std::unique_ptr<DecisionFunctionBase>(new SigmoidDecisionFunction(coefficients, support_vectors, bias, gamma, r));
    }

    return decision_function;
  }

  std::unique_ptr<BinarySVC> PraseBinaryClassifier(const YAML::Node& data, const YAML::Node& kernel, bool inverse = false) {
    const float k = (inverse) ? -1.0f : 1.0f;

    std::unique_ptr<BinarySVC> classifier = nullptr;

    auto decision_function = ::ParseDecisionFunction(data, kernel, inverse);

    if (data["platt-probability"]) {
      const auto a = data["platt-probability"]["a"].as<float>(); // not inverting because decision function is already inverted at this point
      const auto b = k * data["platt-probability"]["b"].as<float>();

      classifier = std::unique_ptr<BinarySVC>(new PlattBinarySVC(std::move(decision_function), a, b));
    } else {
      classifier = std::unique_ptr<BinarySVC>(new BinarySVC(std::move(decision_function)));
    }

    return classifier;
  }

  std::unique_ptr<ClassifierBase> ParseOVOClassifier(const YAML::Node& data) {
    const auto kernel = data["kernel"];
    const auto labels = data["labels"].as<std::vector<std::string>>();

    std::unordered_map<std::string, size_t> label_index{};
    for (size_t i = 0; i < labels.size(); i++) {
      label_index[labels[i]] = i;
    }

    std::vector<std::unique_ptr<BinarySVC>> classifiers(labels.size() * (labels.size() - 1) / 2);

    for (const auto classifier_node : data["binary-classifiers"]) {
      auto false_index = label_index[classifier_node["false-label"].as<std::string>()];
      auto true_index = label_index[classifier_node["true-label"].as<std::string>()];

      bool inverse = false;
      if (false_index > true_index) {
        std::swap(false_index, true_index);
        inverse = true;
      }

      const auto classifier_index = utility::pair_index(false_index, true_index, labels.size());
      classifiers[classifier_index] = ::PraseBinaryClassifier(classifier_node, kernel, inverse);
    }

    if (labels.size() == 2) return std::unique_ptr<ClassifierBase>(std::move(classifiers[0]));

    return std::unique_ptr<ClassifierBase>(new OVOClassifier(std::move(classifiers)));
  }

  std::unique_ptr<ClassifierBase> ParseOVRClassifier(const YAML::Node& data) {
    const auto kernel = data["kernel"];
    const auto labels = data["labels"].as<std::vector<std::string>>(); // TODO: throw in case of 2 labels because this case should be exported as OVO

    std::unordered_map<std::string, size_t> label_index{};
    for (size_t i = 0; i < labels.size(); i++) {
      label_index[labels[i]] = i;
    }

    std::vector<std::unique_ptr<BinarySVC>> classifiers(labels.size());

    for (const auto classifier_node : data["binary-classifiers"]) {
      const auto classifier_index = label_index[classifier_node["true-label"].as<std::string>()];
      classifiers[classifier_index] = ::PraseBinaryClassifier(classifier_node, kernel);
    }

    return std::unique_ptr<ClassifierBase>(new OVRClassifier(std::move(classifiers)));
  }

  std::unique_ptr<ClassifierBase> ParseClassifier(const YAML::Node& data, bool with_checks = false) {
    const auto version = data["version"].as<std::string>();
    if (version != "1.0") throw std::runtime_error("Invalid version");

    const auto svc = data["svc"];
    const auto multiclass_strategy = svc["multiclass-strategy"].as<std::string>();

    std::unique_ptr<ClassifierBase> classifier = nullptr;

    if (multiclass_strategy == "one-vs-one")
      classifier = ::ParseOVOClassifier(svc);
    else
      classifier = ::ParseOVRClassifier(svc);

    if (with_checks && svc["sanity-checks"]) {
      const auto labels = svc["labels"].as<std::vector<std::string>>();

      std::unordered_map<std::string, size_t> label_index{};
      for (size_t i = 0; i < labels.size(); i++) {
        label_index[labels[i]] = i;
      }

      for (const auto check_node : svc["sanity-checks"]) {
        const auto features = check_node["features"].as<std::vector<float>>();
        const auto correct_index = label_index[check_node["label"].as<std::string>()];
        const auto predicted_index = classifier->PredictLabel(features);

        if (correct_index != predicted_index) throw std::runtime_error("Sanity check didn't pass");

        if (check_node["probabilities"]) {
          const auto correct_probabilities = check_node["probabilities"].as<std::vector<float>>();
          const auto predicted_probabilities = classifier->PredictProbability(features);

          if (!utility::equal_approx(correct_probabilities, predicted_probabilities)) throw std::runtime_error("Sanity check didn't pass");
        }
      }
    }

    return classifier;
  }

  SVR ParseRegressor(const YAML::Node& data, bool with_checks = false) {
    // TODO: proper schema validation
    const auto version = data["version"].as<std::string>();
    if (version != "1.0") throw std::runtime_error("Invalid version");

    const auto svr = data["svr"];
    const auto kernel = svr["kernel"];

    SVR regressor(::ParseDecisionFunction(svr, kernel));

    if (with_checks && svr["sanity-checks"]) {
      for (const auto check_node : svr["sanity-checks"]) {
        const auto features = check_node["features"].as<std::vector<float>>();
        const auto correct_value = check_node["value"].as<float>();
        const auto predicted_value = regressor.Predict(features);

        if (!utility::equal_approx(correct_value, predicted_value)) throw std::runtime_error("Sanity check didn't pass");
      }
    }

    return regressor;
  }
} // namespace

namespace neonsvm {
  std::unique_ptr<ClassifierBase> LoadClassifier(const char* data, bool with_checks) {
    return ::ParseClassifier(YAML::Load(data), with_checks);
  }

  std::unique_ptr<ClassifierBase> LoadClassifierFile(const char* filename, bool with_checks) {
    return ::ParseClassifier(YAML::LoadFile(filename), with_checks);
  }

  SVR LoadRegressor(const char* data, bool with_checks) {
    return ::ParseRegressor(YAML::Load(data), with_checks);
  }

  SVR LoadRegressorFile(const char* filename, bool with_checks) {
    return ::ParseRegressor(YAML::LoadFile(filename), with_checks);
  }
} // namespace neonsvm
