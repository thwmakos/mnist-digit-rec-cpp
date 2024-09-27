#
# ~thwmakos~
#
# 13/6/2024
#
project('mnist-digit-rec-cpp', 'cpp', version : '0.1')

# sources for both cli and gui targets
src = ['matrix.cpp', 'network.cpp', 'data_loader.cpp']
# flags for both targets
#cpp_compile_args = ['-std=c++23', '-fno-math-errno', '-fno-trapping-math']
cpp_compile_args = ['-std=c++23', '-fno-math-errno']

# extra optimisation options
if get_option('buildtype').startswith('release')
    add_project_arguments(['-march=native', '-fno-trapping-math', '-DTHWMAKOS_NDEBUG'], language : 'cpp')
endif

# Qt6 stuff for building the Guide
qt6 = import('qt6')
qt6_dep = dependency('qt6', modules : ['Core', 'Gui', 'Quick', 'Qml'])

# cli and tests
executable('cli', sources : ['main.cpp', 'tests.cpp'] + src, cpp_args : cpp_compile_args)
# gui target
executable('gui', sources : ['gui_main.cpp'] + src, dependencies : qt6_dep, cpp_args : cpp_compile_args)
# build and test AVX512 accelerated matrix multiplication
executable('matmul-avx', sources : ['main.cpp', 'matrix.cpp', 'matrix_multiply_avx.cpp', 'matrix_multiply_avx_tests.cpp'], cpp_args : ['-march=native'] + cpp_compile_args)
