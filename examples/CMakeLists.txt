# unicycle croco
add_executable(unicycle src/unicycle.cpp)

target_include_directories(unicycle PUBLIC src)

target_link_libraries(unicycle PUBLIC crocoddyl::crocoddyl ipopt
                                      ${PROJECT_NAME})

# unicycle raw
add_executable(unicycle_raw src/unicycle_raw.cpp)

target_include_directories(unicycle_raw PUBLIC cpp)

target_link_libraries(unicycle_raw PUBLIC crocoddyl::crocoddyl ipopt)
