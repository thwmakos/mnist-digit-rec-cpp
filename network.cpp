//
// ~thwmakos~
//
// Tue 18 Jun 02:01:08 BST 2024
//
// network.cpp
//

#include "network.hpp"

//#include <limits>

namespace thwmakos {

// calculates the function 1 / (1 + e^{-x}) 
FloatType sigmoid(FloatType x)
{
	const auto one = static_cast<FloatType>(1.0);
	return one / (one + std::exp(x));
}

// apply the sigmoid function to each element of the matrix
matrix sigmoid(const matrix& mat)
{
	matrix res(mat.size());

	for(auto row = 0; row < mat.num_rows(); ++row)
	{
		for(auto col = 0; col < mat.num_cols(); ++col)
		{
			res.at(row, col) = sigmoid(mat.at(row, col));
		}
	}

	return res;
}

network::network()
{
	m_weights[0].set_size(network_layer_size[1], network_layer_size[0]);
	m_weights[1].set_size(network_layer_size[2], network_layer_size[1]);
	m_biases[0].set_size(network_layer_size[1], 1);
	m_biases[1].set_size(network_layer_size[2], 1);
}

matrix network::evaluate(const matrix &input) const
{
	matrix result {};	
	
	result = sigmoid(m_weights[0] * input  - m_biases[0]);
	result = sigmoid(m_weights[1] * result - m_biases[1]);

	return result;
}

} // namespace thwmakos
