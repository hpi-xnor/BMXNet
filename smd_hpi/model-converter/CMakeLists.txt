include_directories(../libs/)
add_executable(model-converter main.cpp)
target_link_libraries(model-converter mxnet)
