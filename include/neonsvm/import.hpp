#ifndef _NEONSVM_IMPORT_HPP_
#define _NEONSVM_IMPORT_HPP_

#include "neonsvm/classification/base.hpp"
#include "neonsvm/regression/svr.hpp"
#include <memory>

namespace neonsvm {
  std::unique_ptr<ClassifierBase> LoadClassifier(const char* data, bool with_checks = false);
  std::unique_ptr<ClassifierBase> LoadClassifierFile(const char* filename, bool with_checks = false);

  SVR LoadRegressor(const char* data, bool with_checks = false);
  SVR LoadRegressorFile(const char* filename, bool with_checks = false);
} // namespace neonsvm

#endif
