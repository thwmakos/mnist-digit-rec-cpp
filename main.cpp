//
// ~thwmakos~
//
// 13/6/2024
//

#include <iostream>

// doctest testing library
#ifndef NDEBUG
#define DOCTEST_CONFIG_SUPER_FAST_ASSERTS
#define DOCTEST_CONFIG_IMPLEMENT
#else
#define DOCTEST_CONFIG_DISABLE
#endif

// uncomment to disable running of tests
//#define DOCTEST_CONFIG_DISABLE
#include <doctest/doctest.h>


int main(int argc, char *argv[])
{
	doctest::Context ctx;
	
	ctx.setOption("abort-after", 5);  // stop after 5 failed asserts
	ctx.setOption("no-break", true);  // don't break if debugging and a test case fails
	ctx.applyCommandLine(argc, argv);
	
	auto res = ctx.run(); // run test cases unless --no-run

	return res;
}
