# mnist-digit-rec-cpp

Implementation of the classic neural network that recognises
handwritten digits, trained on the MNIST dataset in pure C++23
avoiding external dependencies as much as possible. For personal
research and learning, both theoretical and programming aspects.

Training and test data are located in the `/data` directory.

## Required to build (so far):
- C++23
- `doctest` unit testing library

## Building
- `meson setup builddir && cd builddir && meson compile`
- to run tests, `./cli`
