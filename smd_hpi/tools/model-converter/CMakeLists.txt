include_directories(../../lib/)
add_executable(model-converter main.cpp)
target_link_libraries(model-converter mxnet)
