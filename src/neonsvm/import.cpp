#include "neonsvm/import.hpp"
#include "neonsvm/utility.hpp"

namespace neonsvm {
  std::unique_ptr<ClassifierBase> load_classifier(const char* data) {
    NOT_IMPLEMENTED;
  }

  std::unique_ptr<ClassifierBase> load_classifier_from_file(const char* filename) {
    NOT_IMPLEMENTED;
  }

  SVR load_regressor(const char* data) {
    NOT_IMPLEMENTED;
  }

  SVR load_regressor_from_file(const char* filename) {
    NOT_IMPLEMENTED;
  }
} // namespace neonsvm
