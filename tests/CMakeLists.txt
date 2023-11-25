add_executable(transformer_test transformer_test.cpp)

target_link_libraries(transformer_test PRIVATE armadillo)

# Configure the output path for the executables
set_target_properties(transformer_test PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")

