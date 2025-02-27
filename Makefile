.PHONY: all install  build test lint-check format-check format clean

SRC_FILES := $(shell find src tst -name '*.cpp' -o -name '*.hpp')

all: build

install:
	conan install . --build=missing

build: install
	mkdir -p build
	cd build && cmake .. -DCMAKE_TOOLCHAIN_FILE=Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release -G Ninja
	cd build && cmake --build . -j

test: build
	cd build && ./tree_tests

benchmark: build
	cd build && ./tree_benchmark

lint-check:
	clang-tidy $(SRC_FILES) -p build

format-check:
	clang-format --style=file --Werror --dry-run $(SRC_FILES)

format:
	clang-format --style=file -i $(SRC_FILES)
	clang-tidy -fix -p build $(SRC_FILES)

clean:
	rm -rf build
