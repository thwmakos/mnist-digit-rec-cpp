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
FloatType sigmoid(FloatType x);

// apply the sigmoid function to each element of the matrix
matrix sigmoid(const matrix& mat);

// neural network layers determined at compile time
// input and output layers are first and last elements
// of the array, so they are included here
constexpr std::array<FloatType, 3> network_layer_size = {28 * 28, 30, 10};

struct network
{
	// default constructor, zero-initialises all the weights and biases
	explicit network();

	// evaluate the network
	// takes a column vector whose length must match the length of the input layer
	// returns a vector representing the activation of the final layer
	matrix evaluate(const matrix &input) const;

	// weight matrices for every layer except first one
	// the weights of a given neuron in layer are represented by a 
	// matrix row, with one entry for every neuron in the previous layer 
	std::array<matrix, network_layer_size.size() - 1> m_weights;
	
	// column vectors for biases
	// first layer neurons do not have biases
	std::array<matrix, network_layer_size.size() - 1> m_biases;

};

} // namespace thwmakos

#endif // NETWORK_HPP_INCLUDED
