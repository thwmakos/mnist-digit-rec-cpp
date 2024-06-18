//
// ~thwmakos~
//
// Mon 17 Jun 23:15:59 BST 2024
//
// network.hpp
//

#ifndef NETWORK_HPP_INCLUDED
#define NETWORK_HPP_INCLUDED

#include "matrix.hpp"

#include <array>
#include <cmath>

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

// neural network layers determined at compile time
// input and output layers are first and last elements
// of the array, so they are included here
constexpr std::array<FloatType, 3> network_layer_size = {28 * 28, 30, 10};

struct network
{
	network()
	{
		m_weights[0].set_size(network_layer_size[1], network_layer_size[0]);
		m_weights[1].set_size(network_layer_size[2], network_layer_size[1]);
		m_biases[0].set_size(network_layer_size[1], 1);
		m_biases[1].set_size(network_layer_size[2], 1);
	}

	// weight matrices for every layer except first one
	// the weights of a given neuron in layer are represented by a 
	// matrix row, with one entry for every neuron in the previous layer 
	std::array<matrix, network_layer_size.size() - 1> m_weights;
	
	// column vectors for biases
	// first layer neurons do not have biases
	std::array<matrix, network_layer_size.size() - 1> m_biases;

	matrix evaluate(const matrix &input) const
	{
		matrix result {};	
		
		result = sigmoid(m_weights[0] * input  - m_biases[0]);
		result = sigmoid(m_weights[1] * result - m_biases[1]);

		return result;
	}
};

} // namespace thwmakos

#endif // NETWORK_HPP_INCLUDED
