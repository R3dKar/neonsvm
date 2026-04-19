cmake_minimum_required(VERSION 3.10)
project(neonsvm VERSION 0.1.0 LANGUAGES CXX)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests executable" ${PROJECT_IS_TOP_LEVEL})

include(CheckCXXSourceCompiles)
check_cxx_source_compiles([[
  #include <arm_neon.h>
  int main() {
    float32x4_t v = {1.2, 3.4, 5.6, 7.8};
    v = vfmaq_f32(v, v, v);
    return 0;
  }
]] HAVE_NEON)
if (NOT HAVE_NEON)
  message(FATAL_ERROR "NEON SIMD is not supported")
endif()

include(FetchContent)

set(YAML_CPP_BUILD_TOOLS OFF)
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.9.0
)
FetchContent_MakeAvailable(yaml-cpp)

if (BUILD_TESTS)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.17.0
  )
  FetchContent_MakeAvailable(googletest)
endif()

file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS src/*.cpp)

if (BUILD_SHARED_LIBS)
  add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCES})
else()
  add_library(${PROJECT_NAME} STATIC ${PROJECT_SOURCES})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC include PRIVATE src)
target_link_libraries(${PROJECT_NAME} PUBLIC yaml-cpp::yaml-cpp)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_11)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_EXTENSIONS OFF)

if (BUILD_TESTS)
  enable_testing()

  add_executable(${PROJECT_NAME}_test test/main.cpp)
  target_link_libraries(${PROJECT_NAME}_test PRIVATE ${PROJECT_NAME} GTest::gtest_main)

  include(GoogleTest)
  gtest_discover_tests(${PROJECT_NAME}_test)

  add_custom_command(TARGET ${PROJECT_NAME}_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/test/models/ $<TARGET_FILE_DIR:${PROJECT_NAME}_test>/models/
  )
endif()
