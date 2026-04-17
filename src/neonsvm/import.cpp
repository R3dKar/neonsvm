#include "neonsvm/import.hpp"
#include "neonsvm/decision_function/base.hpp"
#include "neonsvm/decision_function/linear.hpp"
#include "neonsvm/decision_function/polynomial.hpp"
#include "neonsvm/decision_function/rbf.hpp"
#include "neonsvm/decision_function/sigmoid.hpp"
#include "neonsvm/regression/svr.hpp"
#include "neonsvm/utility.hpp"
#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h>

namespace {
  using namespace neonsvm;

  std::unique_ptr<ClassifierBase> ParseClassifier(const YAML::Node& data, bool ignore_checks) {
    NOT_IMPLEMENTED;
  }

  SVR ParseRegressor(const YAML::Node& data, bool ignore_checks) {
    const std::string version = data["version"].as<std::string>();
    if (version != "1.0") throw std::runtime_error("Invalid version");

    const YAML::Node svr = data["svr"];

    const float bias = svr["bias"].as<float>();

    std::vector<std::vector<float>> support_vectors{};
    support_vectors.reserve(svr["support-vectors"].size());

    std::vector<float> coefficients{};
    coefficients.reserve(svr["support-vectors"].size());

    for (const auto vector_node : svr["support-vectors"]) {
      const std::vector<float> support_vector = vector_node["vector"].as<std::vector<float>>();
      const float coefficient = vector_node["coefficient"].as<float>(1.0f);

      support_vectors.emplace_back(std::move(support_vector));
      coefficients.push_back(coefficient);
    }

    std::unique_ptr<DecisionFunctionBase> decision_function = nullptr;

    const YAML::Node kernel = svr["kernel"];
    const std::string kernel_type = kernel["type"].as<std::string>();
    if (kernel_type == "linear") {
      decision_function = std::unique_ptr<DecisionFunctionBase>(new LinearDecisionFunction(coefficients, support_vectors, bias));

    } else if (kernel_type == "polynomial") {
      const float gamma = kernel["gamma"].as<float>();
      const float r = kernel["bias"].as<float>();
      const uint32_t degree = kernel["degree"].as<uint32_t>();
      decision_function =
          std::unique_ptr<DecisionFunctionBase>(new PolynomialDecisionFunction(coefficients, support_vectors, bias, gamma, r, degree));

    } else if (kernel_type == "rbf") {
      const float gamma = kernel["gamma"].as<float>();
      decision_function = std::unique_ptr<DecisionFunctionBase>(new RBFDecisionFunction(coefficients, support_vectors, bias, gamma));

    } else if (kernel_type == "sigmoid") {
      const float gamma = kernel["gamma"].as<float>();
      const float r = kernel["bias"].as<float>();
      decision_function = std::unique_ptr<DecisionFunctionBase>(new SigmoidDecisionFunction(coefficients, support_vectors, bias, gamma, r));
    }

    SVR regressor(std::move(decision_function));

    if (!ignore_checks && svr["sanity-checks"]) {
      for (const auto check_node : svr["sanity-checks"]) {
        const std::vector<float> features = check_node["features"].as<std::vector<float>>();
        const float value = check_node["value"].as<float>();

        const float predicted_value = regressor.Predict(features);

        if (!utility::equal_approx(value, predicted_value)) throw std::runtime_error("Sanity check didn't pass");
      }
    }

    return regressor;
  }
} // namespace

namespace neonsvm {
  std::unique_ptr<ClassifierBase> LoadClassifier(const char* data, bool ignore_checks) {
    return ::ParseClassifier(YAML::Load(data), ignore_checks);
  }

  std::unique_ptr<ClassifierBase> LoadClassifierFile(const char* filename, bool ignore_checks) {
    return ::ParseClassifier(YAML::LoadFile(filename), ignore_checks);
  }

  SVR LoadRegressor(const char* data, bool ignore_checks) {
    return ::ParseRegressor(YAML::Load(data), ignore_checks);
  }

  SVR LoadRegressorFile(const char* filename, bool ignore_checks) {
    return ::ParseRegressor(YAML::LoadFile(filename), ignore_checks);
  }
} // namespace neonsvm
