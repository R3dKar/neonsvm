#ifndef _NEONSVM_IMPORT_HPP_
#define _NEONSVM_IMPORT_HPP_

#include "neonsvm/classification/base.hpp"
#include "neonsvm/regression/svr.hpp"
#include <memory>

namespace neonsvm {
  std::unique_ptr<ClassifierBase> load_classifier(const char* data);
  std::unique_ptr<ClassifierBase> load_classifier_from_file(const char* filename);

  SVR load_regressor(const char* data);
  SVR load_regressor_from_file(const char* filename);
} // namespace neonsvm

#endif
