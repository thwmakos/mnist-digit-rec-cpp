#
# ~thwmakos~
#
# 13/6/2024
#
project('mnist-digit-rec-cpp', 'cpp', version : '0.2')

# sources for both cli and gui targets
src = ['matrix.cpp', 'network.cpp', 'data_loader.cpp', 'matrix_avx.cpp']
# flags for both targets
cpp_compile_args = ['-std=c++23', '-march=native', '-fopenmp']

# extra optimisation options
if get_option('buildtype').startswith('release')
    add_project_arguments(['-ffast-math', '-DNDEBUG'], language : 'cpp')
endif

# for parallel stl algorithms
tbb = dependency('tbb')

# Qt6 stuff for building the Guide
#qt6 = import('qt6')
qt6_dep = dependency('qt6', modules : ['Core', 'Gui', 'Quick', 'Qml'], required : false)

# cli target
executable('cli', sources : ['cli_main.cpp'] + src, dependencies : tbb, cpp_args : cpp_compile_args)
# tests target
executable('tests', sources : ['tests_main.cpp', 'tests.cpp', 'matrix_avx_tests.cpp'] + src, dependencies : tbb, cpp_args : cpp_compile_args)
# gui target
executable('gui', sources : ['gui_main.cpp'] + src, dependencies : qt6_dep, cpp_args : cpp_compile_args)
# build and test AVX512 accelerated matrix multiplication
