cmake_minimum_required(VERSION 3.10)
project(neonsvm VERSION 0.1.0 LANGUAGES CXX)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

include(FetchContent)

set(YAML_CPP_BUILD_TOOLS OFF)
FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
  GIT_TAG yaml-cpp-0.9.0
)
FetchContent_MakeAvailable(yaml-cpp)


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
