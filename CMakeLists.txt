cmake_minimum_required(VERSION 3.20)

project(tree-race LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(DEBUG_FLAGS -g -O0)
set(RELEASE_FLAGS -O3 -DNDEBUG)

find_package(GTest REQUIRED)
enable_testing()

file(GLOB MAIN src/main.cpp)
file(GLOB_RECURSE TESTS tst/*.cpp)
file(GLOB_RECURSE HEADERS src/*.hpp)

find_package(fmt REQUIRED)

SET(PACKAGES fmt::fmt)

add_executable(tree_exe ${MAIN} ${HEADERS})
add_executable(tree_tests ${TESTS} ${HEADERS})
add_executable(tree_debug ${MAIN} ${HEADERS})

target_compile_options(tree_exe PRIVATE ${RELEASE_FLAGS})
target_compile_options(tree_tests PRIVATE ${DEBUG_FLAGS})
target_compile_options(tree_debug PRIVATE ${DEBUG_FLAGS})


SET(TARGETS tree_exe tree_tests tree_debug)

foreach (target ${TARGETS})
    target_include_directories(${target} PUBLIC src)
    target_link_libraries(${target} PRIVATE ${PACKAGES})
endforeach()

# Gtest
target_link_libraries(tree_tests PRIVATE gtest::gtest)
