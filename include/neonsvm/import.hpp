#ifndef _NEONSVM_IMPORT_HPP_
#define _NEONSVM_IMPORT_HPP_

#include "neonsvm/classification/base.hpp"
#include "neonsvm/regression/svr.hpp"
#include <memory>

namespace neonsvm {
  /**
   * @brief Loads classifier from `data` YAML string.
   *
   * Parses `data` YAML string and returns a pointer to an object that implemnets `ClassifierBase` abstract class. Runs model checks if specified so.
   *
   * @param data Model's data YAML string to be parsed.
   * @param with_checks If true, runs checks that included in `data`. If no checks found, this parameter is ignored.
   * @return Parsed classifier object.
   * @throws `std::runtime_error` thrown if checks are failed.
   */
  std::unique_ptr<ClassifierBase> LoadClassifier(const char* data, bool with_checks = false);

  /**
   * @brief Loads classifier from YAML file at `filename`.
   *
   * Parses YAML file at `filename` and returns a pointer to an object that implements `ClassifierBase` abstract class. Runs model checks if specified
   * so.
   *
   * @param filename Path to model YAML file.
   * @param with_checks If true, runs checks that included in file. If no checks found, this parameter is ignored.
   * @return Parsed classifier object.
   * @throws `std::runtime_error` thrown if checks are failed.
   */
  std::unique_ptr<ClassifierBase> LoadClassifierFile(const char* filename, bool with_checks = false);

  /**
   * @brief Loads regressor from `data` YAML string.
   *
   * Parses `data` YAML string and returns `SVR` object. Runs model checks if specified so.
   *
   * @param data Model's data YAML string to be parsed.
   * @param with_checks If true, runs checks that included in `data`. If no checks found, this parameter is ignored.
   * @return Parsed regressor object.
   * @throws `std::runtime_error` thrown if checks are failed.
   */
  SVR LoadRegressor(const char* data, bool with_checks = false);

  /**
   * @brief Loads regressor from YAML file at `filename`.
   *
   * Parses YAML file at `filename` and returns `SVR` object. Runs model checks if specified so.
   *
   * @param filename Path to model YAML file.
   * @param with_checks If true, runs checks that included in file. If no checks found, this parameter is ignored.
   * @return Parsed regressor object.
   * @throws `std::runtime_error` thrown if checks are failed.
   */
  SVR LoadRegressorFile(const char* filename, bool with_checks = false);
} // namespace neonsvm

#endif
