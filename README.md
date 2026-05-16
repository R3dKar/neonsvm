# neonsvm

Library for optimized inference SVM-based classification and regression models on ARM devices with NEON SIMD.

## Usage

Train model and export with [`skl2neonsvm`](https://github.com/R3dKar/sklearn-neonsvm) python library:

```python
# Training model
from sklearn.datasets import make_classification
from sklearn.svm import SVC

X, y = make_classification(random_state=42)
model = SVC().fit(X, y)

# Converting and exporting
from skl2neonsvm import to_neonsvm

data = to_neonsvm(model)
with open("model.yaml", "w") as f:
    f.write(data)
```

Then import and run inference:

```cpp
#include <iostream>
#include <neonsvm/neonsvm.hpp>

int main() {
  const auto classifier = neonsvm::LoadClassifierFile("./model.yaml");

  std::cout << classifier->PredictLabel({1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0}) << '\n';

  return 0;
}
```

There are following functions can be used for importing models:

```cpp
// Load classifier
std::unique_ptr<ClassifierBase> LoadClassifier(const char* data, bool with_checks = false); // from string
std::unique_ptr<ClassifierBase> LoadClassifierFile(const char* filename, bool with_checks = false); // from file

// Load regressor
SVR neonsvm::LoadRegressor(const char* data, bool with_checks = false); // from string
SVR neonsvm::LoadRegressorFile(const char* filename, bool with_checks = false); // from file
```

## Build

Build process is default as for most cmake projects:

```bash
git clone https://github.com/R3dKar/neonsvm.git
cd neonsvm
mkdir build
cmake -S . -B build
cmake --build build
```

Also you can include library using `FetchContent`:

```cmake
include(FetchContent)

FetchContent_Declare(
  neonsvm
  GIT_REPOSITORY https://github.com/R3dKar/neonsvm.git
  GIT_TAG v0.1.1
)
FetchContent_MakeAvailable(neonsvm)

...

target_link_library(my_app PRIVATE neonsvm)
```
