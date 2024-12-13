#
# ~thwmakos~
#
# 13/6/2024
#
project('mnist-digit-rec-cpp', 'cpp', version : '0.2',
    default_options : ['cpp_std=c++23', 'warning_level=3', 'b_ndebug=if-release'])

# sources for both cli and gui targets
src = ['source/matrix.cpp', 'source/network.cpp', 'source/data_loader.cpp', 'source/matrix_avx.cpp']
# flags for both targets
cpp_compile_args = ['-march=native']

# extra optimisation options
if get_option('buildtype').startswith('release')
    add_project_arguments(['-ffast-math', '-DNDEBUG'], language : 'cpp')
endif

# for parallel stl algorithms
tbb = dependency('tbb')

# Qt6 stuff for building the Guide
#qt6 = import('qt6')
qt6_dep = dependency('qt6', modules : ['Core', 'Gui', 'Widgets'], required : false)

# cli target
executable('cli', sources : ['source/cli_main.cpp'] + src, dependencies : tbb, cpp_args : cpp_compile_args)
# tests target
executable('tests', sources : ['test/tests_main.cpp', 'test/tests.cpp', 'test/matrix_avx_tests.cpp'] + src, dependencies : tbb, cpp_args : cpp_compile_args)
# gui target
executable('gui', sources : ['source/gui_main.cpp'] + src, dependencies : [qt6_dep, tbb], cpp_args : cpp_compile_args)
# build and test AVX512 accelerated matrix multiplication
