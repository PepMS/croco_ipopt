cmake_minimum_required(VERSION 3.12)
project(
  ipopt_croco
  VERSION 0.0.0
  LANGUAGES CXX)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

find_package(crocoddyl REQUIRED)
find_package(ipopt REQUIRED)

set(Python_ADDITIONAL_VERSIONS 3.6 3.7 3.8)
find_package(Python3 REQUIRED Development Interpreter)
set(WHICH_PYTHON3
    "python3${Python3_VERSION_MINOR}"
    CACHE STRING "Python3 version")
find_package(
  Boost
  COMPONENTS ${WHICH_PYTHON3}
  REQUIRED)

# Library
add_library(${PROJECT_NAME} SHARED src/multiple-shooting-nlp.cpp src/ipopt.cpp)

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_definitions(${PROJECT_NAME} PRIVATE cxx_std_14)

target_link_libraries(${PROJECT_NAME} PUBLIC crocoddyl::crocoddyl ipopt)

# Examples
add_subdirectory(examples)

# Bindings
add_subdirectory(bindings)
