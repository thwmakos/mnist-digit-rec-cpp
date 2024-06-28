//
// ~thwmakos~
//
// Tue 18 Jun 02:01:08 BST 2024
//
// network.cpp
//

#include "network.hpp"
#include "data_loader.hpp"

#include <cmath>
#include <random>

namespace thwmakos {

// calculates the function 1 / (1 + e^{-x})
// TODO: add test for this function 
FloatType sigmoid(FloatType x)
{
	const auto one = static_cast<FloatType>(1.0);
	return one / (one + std::exp(-x));
}

// calculates derivative of sigmoid
// TODO: add test for this function too
FloatType sigmoid_derivative(FloatType x)
{
	const auto one = static_cast<FloatType>(1.0);	
	const auto exp_minus_x = std::exp(-x);

	return exp_minus_x / ((one + exp_minus_x) * (one + exp_minus_x));
}

network::network()
{
	// allocate space for the weight and bias matrices
	m_weights[0].set_size(network_layer_size[1], network_layer_size[0]);
	m_weights[1].set_size(network_layer_size[2], network_layer_size[1]);
	m_biases[0].set_size(network_layer_size[1], 1);
	m_biases[1].set_size(network_layer_size[2], 1);
	
	// initialise matrices with random normally distributed entries
	std::random_device rd {}; 
	std::default_random_engine eng { rd() };
	std::normal_distribution<FloatType> normal(0.0f, 1.0f);
	
	// lambda to randomise an sequence of matrices using 
	// the random distribution constructed above
	auto randomise = [&normal, &eng] (auto& matrices)
	{
		for(auto&& mat : matrices)
		{
			const auto [num_rows, num_cols] = mat.size();

			for(auto i = 0; i < num_rows; ++i)
			{
				for(auto j = 0; j < num_cols; ++j)	
				{
					mat[i, j] = normal(eng);
				}
			}
		}
	};

	randomise(m_weights);
	randomise(m_biases);
}

matrix network::evaluate(const matrix &input) const
{
	// check if input size is correct
	// has to be equal to the number of input layers
	if(input.num_rows() != m_weights[0].num_cols() ||
		input.num_cols() != 1)
	{
		throw std::invalid_argument("expected column vector with length of input layer"); 
	}

	matrix result {};	
		
	result = elementwise_apply(m_weights[0] * input  - m_biases[0], [](FloatType x){return 1.0f; });
	result = elementwise_apply(m_weights[1] * result - m_biases[1], sigmoid);

	return result;
}

void network::train()
{
	data_loader dl{"../data/train-images-idx3-ubyte", "./data/train-labels-idx1-ubyte"};
}

} // namespace thwmakos
