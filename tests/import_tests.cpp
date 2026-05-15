#include <gtest/gtest.h>
#include <neonsvm/neonsvm.hpp>
#include <string>

using namespace std::string_literals;

struct ClassifierModelFileTest : testing::TestWithParam<const char*> {};

TEST_P(ClassifierModelFileTest, LoadClassifierWithChecks) {
  const auto model_file = "./models/classification/"s + GetParam();
  const auto model = neonsvm::LoadClassifierFile(model_file.c_str(), true);

  EXPECT_NE(model, nullptr);
}

INSTANTIATE_TEST_SUITE_P(ClassificationTests, ClassifierModelFileTest, testing::Values(
  "svc_2_linear.yaml",
  "svc_2_poly.yaml",
  "svc_2_rbf.yaml",
  "svc_2_sigmoid.yaml",
  "svc_5_linear_ovo.yaml",
  "svc_5_linear_ovr.yaml",
  "svc_5_poly_ovo.yaml",
  "svc_5_rbf_ovo.yaml",
  "svc_5_sigmoid_ovo.yaml"
));

struct RegressionModelFileTest : testing::TestWithParam<const char*> {};

TEST_P(RegressionModelFileTest, LoadRegressorWithChecks) {
  const auto model_file = "./models/regression/"s + GetParam();
  const auto model = neonsvm::LoadRegressorFile(model_file.c_str(), true);
}

INSTANTIATE_TEST_SUITE_P(RegressionTests, RegressionModelFileTest, testing::Values(
  "svr_linear.yaml",
  "svr_poly.yaml",
  "svr_rbf.yaml",
  "svr_sigmoid.yaml"
));
