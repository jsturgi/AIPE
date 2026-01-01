cmake_minimum_required(VERSION 3.14)
project(LinearAlgebraVectors)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Google Test
find_package(GTest REQUIRED)

# Create test executable
add_executable(vector_test
    tests/vector_test.cpp
)

# Link against Google Test
target_link_libraries(vector_test
    GTest::GTest
    GTest::Main
)

# Include src directory for headers
target_include_directories(vector_test PRIVATE src)

# Enable testing
enable_testing()
add_test(NAME vector_test COMMAND vector_test)
