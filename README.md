# Fastest ID3 Decision Tree

This project is the result of a club competition to develop the fastest possible implementation of the ID3 decision tree algorithm. The goal was to optimize both the training (tree building) and inference (prediction) phases.

## Features

* Optimized ID3 algorithm implementation in C++.
* Uses Google Benchmark for performance measurement.
* Includes test cases using Google Test.
* Uses `csv.h` for efficient CSV data loading.

## Building

This project uses [Conan](https://conan.io/) for dependency management and CMake for building.

1.  **Install Conan:** Follow the instructions on the [Conan website](https://conan.io/downloads).
2.  **Install Dependencies:**
    ```bash
    conan install . --build=missing
    ```
3.  **Build Project:**
    ```bash
    make build
    ```
    This will create a `build` directory and compile the project.

## Running Tests

To ensure the decision tree implementation is correct, run the test suite:

```bash
make test
```

## Running Benchmarks

Performance benchmarks are included using Google Benchmark. To run them:

```bash
make benchmark
```
## Performance Comparison

The primary goal of this project was speed. Below are the benchmark results on the Car Evaluation dataset. For more detailed performance metric, see [perf.md](./perf.md)

| Benchmark      | Our Time  | SciPy time | Speedup |
| -------------- | --------- | ---------- | ------- |
| BM_BuildTree   | 134632 ns | 944930 ns  | 7x      |
| BM_TreePredict | 1091 ns   | 330020 ns  | 302x    |

These results clearly show that this tree significantly outperforms the standard SciPy implementation. While the comparison was attempted to be done as fairly as possible, it is important to note that the SciPy implementation has many more features that are almost certainly not free to include, which likely explains the dramatic difference in inference time in particular.
