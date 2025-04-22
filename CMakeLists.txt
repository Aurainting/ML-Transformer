cmake_minimum_required(VERSION 3.21)
project(ml_transformer LANGUAGES CXX)

find_package(BLAS REQUIRED)

set(Armadillo_DIR "${CMAKE_SOURCE_DIR}/3rdparty/armadillo")
set(Ensmallen_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ensmallen")
set(Doctest_DIR "${CMAKE_SOURCE_DIR}/3rdparty/doctest")

add_subdirectory("${CMAKE_SOURCE_DIR}/test")

add_executable(${PROJECT_NAME} main.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE "${Armadillo_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${Ensmallen_DIR}/include")
target_include_directories(${PROJECT_NAME} PRIVATE "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(${PROJECT_NAME} PRIVATE ${BLAS_LIBRARIES})
