//
// ~thwmakos~
//
// 13/6/2024
//

// doctest testing library
#define DOCTEST_CONFIG_IMPLEMENT

// uncomment to disable running of tests
//#define DOCTEST_CONFIG_DISABLE
#include <doctest/doctest.h>
#include <cfenv> // to enable SIGFPE
#include <print>

#ifdef NDEBUG
	constexpr bool debug = false;
#else
	constexpr bool debug = true;
#endif


int main(int argc, char *argv[])
{
	doctest::Context ctx;
	
	ctx.setOption("abort-after", 5);  // stop after 5 failed asserts
	ctx.setOption("no-break", true);  // don't break if debugging and a test case fails
	ctx.applyCommandLine(argc, argv);

	if constexpr (debug)
	{
		std::println("Debugging: {}", debug);
		// if running in debugger, break on floating point NaN and overflow
		feenableexcept(FE_INVALID | FE_OVERFLOW);
	}

	
	auto res = ctx.run(); // run test cases unless --no-run

	return res;
}
