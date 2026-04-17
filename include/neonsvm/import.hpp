#ifndef _NEONSVM_IMPORT_HPP_
#define _NEONSVM_IMPORT_HPP_

#include "neonsvm/classification/base.hpp"
#include "neonsvm/regression/svr.hpp"
#include <memory>

namespace neonsvm {
  std::unique_ptr<ClassifierBase> LoadClassifier(const char* data, bool ignore_checks = true);
  std::unique_ptr<ClassifierBase> LoadClassifierFile(const char* filename, bool ignore_checks = true);

  SVR LoadRegressor(const char* data, bool ignore_checks = true);
  SVR LoadRegressorFile(const char* filename, bool ignore_checks = true);
} // namespace neonsvm

#endif
