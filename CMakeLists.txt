cmake_minimum_required(VERSION 3.21)
project(ml_transformer LANGUAGES CXX)

find_package(Armadillo CONFIG REQUIRED)

find_path(MLPACK_INCLUDE_DIRS "mlpack/mlpack.hpp")

# Add subdirectory for the test directory
add_subdirectory(tests)

target_include_directories(${PROJECT_NAME} PRIVATE ${MLPACK_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(${PROJECT_NAME} PRIVATE armadillo)

# Configure the output path for the executables
set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin")
