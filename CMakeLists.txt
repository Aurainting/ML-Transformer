cmake_minimum_required(VERSION 3.21)
project(ml_transformer)

find_package(Armadillo CONFIG REQUIRED)

include_directories("${CMAKE_SOURCE_DIR}/include")

# Add subdirectory for the test directory
add_subdirectory(tests)

