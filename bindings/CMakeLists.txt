add_library(${PROJECT_NAME}_pywrap SHARED python/ipopt_croco/ipopt-croco.cpp)

target_include_directories(${PROJECT_NAME}_pywrap PUBLIC  ${Boost_INCLUDE_DIRS} 
                                                          ${Python3_INCLUDE_DIRS}
                                                          ${CMAKE_SOURCE_DIR}/bindings
)

target_link_libraries(${PROJECT_NAME}_pywrap ${Boost_LIBRARIES} 
                                             ${Python3_LIBRARIES}
                                             ${PROJECT_NAME})

target_compile_options(${PROJECT_NAME}_pywrap PRIVATE "-Wno-conversion")