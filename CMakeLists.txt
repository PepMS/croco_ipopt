cmake_minimum_required(VERSION 3.12)
project(
  libcrocoipopt
  VERSION 0.0.0
  LANGUAGES CXX)

list(INSERT CMAKE_MODULE_PATH 0 ${CMAKE_SOURCE_DIR}/cmake)

find_package(crocoddyl REQUIRED)
find_package(ipopt REQUIRED)

# Library

add_library(croco_ipopt src/multiple-shooting-nlp.cpp)

target_include_directories(
  croco_ipopt
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_definitions(croco_ipopt PRIVATE cxx_std_14)

target_link_libraries(croco_ipopt PUBLIC crocoddyl::crocoddyl ipopt)

# unicycle croco
add_executable(unicycle src/unicycle.cpp)

target_include_directories(unicycle PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(unicycle PUBLIC crocoddyl::crocoddyl ipopt croco_ipopt)

# unicycle raw
add_executable(unicycle_raw src/unicycle_raw.cpp)

target_include_directories(unicycle_raw PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(unicycle_raw PUBLIC crocoddyl::crocoddyl ipopt)