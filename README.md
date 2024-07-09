# mnist-digit-rec-cpp

**work in progress**

Implementation of the classic neural network that recognises
handwritten digits, trained on the MNIST dataset in pure C++23
avoiding external dependencies as much as possible. For personal
research and learning, both theoretical and programming aspects.

Training and test data are located in the `/data` directory.

## Required to build (so far):
- C++23
- `doctest` unit testing library
- Qt6 with the `Core, Gui, Quick, Qml` modules for the GUI

## Building
- For testing use the `cli` target: `meson setup builddir && cd builddir && meson compile` and run `./cli`
- For the Qt6 interface use the `gui` target and run `./gui`
