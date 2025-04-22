file(GLOB TEST_SOURCES "${CMAKE_SOURCE_DIR}/test/*.cpp")

add_executable(${PROJECT_NAME}_Test ${TEST_SOURCES})

target_include_directories(${PROJECT_NAME}_Test PRIVATE "${Armadillo_DIR}/include")
target_include_directories(${PROJECT_NAME}_Test PRIVATE "${Ensmallen_DIR}/include")
target_include_directories(${PROJECT_NAME}_Test PRIVATE "${Doctest_DIR}/doctest")
target_include_directories(${PROJECT_NAME}_Test PRIVATE "${CMAKE_SOURCE_DIR}/include")

target_link_libraries(${PROJECT_NAME}_Test PRIVATE ${BLAS_LIBRARIES})
