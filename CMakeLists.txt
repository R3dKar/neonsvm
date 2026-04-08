cmake_minimum_required(VERSION 3.10)
project(neonsvm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build shared libraries" ON)

file(GLOB_RECURSE PROJECT_SOURCES CONFIGURE_DEPENDS src/*.cpp)

if (BUILD_SHARED_LIBS)
  add_library(${PROJECT_NAME} SHARED ${PROJECT_SOURCES})
else()
  add_library(${PROJECT_NAME} STATIC ${PROJECT_SOURCES})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC include PRIVATE src)
