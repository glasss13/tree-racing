cmake_minimum_required(VERSION 3.20)

project(tree-race LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(DEBUG_FLAGS
    -g
    -O0
    -fsanitize=undefined
    -fsanitize=address
    -Wall
    # -Werror
    -Wextra
    -Wpedantic
    -Wconversion
    -Wsign-conversion
    -Wdouble-promotion
    -Wformat=2                   # Enable format warnings
    -Wnull-dereference           # Warn if null pointer dereference is detected
    -Woverloaded-virtual         # Warn if a base class function is hidden
    -Wnon-virtual-dtor           # Warn if a base class has a non-virtual destructor
    -Wold-style-cast             # Warn if using C-style casts
    -Winit-self                  # Warn on self-initialization
    -Wstrict-aliasing=2          # Strong aliasing warnings
    -Wundef
)

set(RELEASE_FLAGS -O3 -DNDEBUG)

find_package(GTest REQUIRED)
enable_testing()

file(GLOB MAIN src/main.cpp)
file(GLOB_RECURSE TESTS tst/*.cpp)
file(GLOB_RECURSE BENCHMARKS bench/*.cpp)
file(GLOB_RECURSE HEADERS src/*.hpp)

find_package(fmt REQUIRED)
find_package(benchmark REQUIRED)

SET(PACKAGES fmt::fmt)

add_executable(tree_exe ${MAIN} ${HEADERS})
add_executable(tree_tests ${TESTS} ${HEADERS})
add_executable(tree_benchmark ${BENCHMARKS} ${HEADERS})
add_executable(tree_debug ${MAIN} ${HEADERS})

target_compile_options(tree_exe PRIVATE ${RELEASE_FLAGS})
target_compile_options(tree_tests PRIVATE ${DEBUG_FLAGS})
target_compile_options(tree_benchmark PRIVATE ${RELEASE_FLAGS})
target_compile_options(tree_debug PRIVATE ${DEBUG_FLAGS})

target_link_libraries(tree_tests PRIVATE -fsanitize=undefined -fsanitize=address)
target_link_libraries(tree_debug PRIVATE -fsanitize=undefined -fsanitize=address)

SET(TARGETS tree_exe tree_tests tree_benchmark tree_debug)

foreach (target ${TARGETS})
    target_include_directories(${target} PUBLIC src)
    target_link_libraries(${target} PRIVATE ${PACKAGES})
endforeach()

# Gtest
target_link_libraries(tree_exe PRIVATE benchmark::benchmark_main)
target_link_libraries(tree_debug PRIVATE benchmark::benchmark_main)
target_link_libraries(tree_tests PRIVATE gtest::gtest)
target_link_libraries(tree_benchmark PRIVATE benchmark::benchmark_main)
