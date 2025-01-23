//
// ~thwmakos~
//
// 21/10/2024
//

#include <print>
#include <iostream>

#include "network.hpp"
#include "data_loader.hpp"

int main()
{
	constexpr std::array layers = { 28 * 28, 30, 10 };

    thwmakos::network nwk { layers };

    nwk.train(15, 1000, 3.0f);

	thwmakos::data_loader loader("../data/t10k-images-idx3-ubyte", "../data/t10k-labels-idx1-ubyte");

	// calculate accuracy
	int correct_evals = 0;

	for(int i = 0; i < loader.m_num_images; ++i)
	{
		auto sample = loader.get_sample(i);
		auto eval   = nwk.evaluate(sample.image);

		if(sample.label_val == output_to_int(eval))
		{
			++correct_evals;	
		}
	}
	
	auto accuracy = static_cast<float>(correct_evals) / static_cast<float>(loader.m_num_images);
	std::println("Got {} out of {} correctly, accuracy is {} %", correct_evals, loader.m_num_images, 100.0f * accuracy);

	// proceed to interactive mode to investigate raw output of specific samples
	while(true)
	{
		int sample_index = -1;
		constexpr auto prompt = "Enter sample index: ";
		std::print("{}", prompt);

		std::cin >> sample_index;
	
		if(!std::cin || sample_index < 0 || sample_index >= loader.m_num_images)
		{
			if(sample_index == -1)
			{
				break;
			}

			std::println("Invalid input or out of bounds");
			continue;
		}

		auto sample = loader.get_sample(sample_index);

		std::println("Sample label is: {}", sample.label_val);
		//std::println("Sample label raw value: {}", sample.label);
		auto eval = nwk.evaluate(sample.image);
		std::println("Network evaluation: {}", output_to_int(eval));
		std::println("Raw network evaluation: {}", eval);
	}

    return 0;
}
