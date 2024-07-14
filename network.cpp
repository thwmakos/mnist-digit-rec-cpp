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
#include <iostream>

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

// returns a column vector with the partial derivatives of the cost function
// in notation this returns \pdv{C_x}{a^L} for the sample (x, y)
// where a^L is output_activations and y is label 
matrix cost_derivative(const matrix& output_activations, const matrix& label)
{
	return output_activations - label;
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

matrix network::evaluate(const matrix& input) const
{
	// check if input size is correct
	// has to be equal to the number of input layers
	if(input.num_rows() != m_weights[0].num_cols() ||
		input.num_cols() != 1)
	{
		throw std::invalid_argument("expected column vector with length of input layer"); 
	}

	matrix result {};	
		
	result = elementwise_apply(m_weights[0] * input  - m_biases[0], sigmoid);
	result = elementwise_apply(m_weights[1] * result - m_biases[1], sigmoid);

	return result;
}

void network::train()
{
	constexpr auto images = "../data/train-images-idx3-ubyte";
	constexpr auto labels = "../data/train-labels-idx1-ubyte";

	data_loader dl(images, labels);
}

network::gradient network::backpropagation(const training_sample& sample) const
{
	network::gradient grad;
	
	// forward pass
	
	// calculate the activations at each layer
	std::vector<matrix> activations; // this is a^l from Nielsen
	std::vector<matrix> weighted_inputs; // this is z^l from Nielsen

	// activation at the first layer is input
	activations.push_back(sample.image);

	// calculate activations and weighted inputs (a^l and z^l) at each layer
	// in notation, we do this: z^l = w^l \cdot a^{l-1} + b^l and a^l = \sigma (z^l)
	for(auto [weight_it, bias_it] = std::tuple { std::cbegin(m_weights), std::cbegin(m_biases) };
			weight_it != std::cend(m_weights); // m_weights and m_biases have the same size
			++weight_it, ++bias_it)
	{
		const matrix& last_activation = *std::prev(std::cend(activations));
		const matrix& weight          = *weight_it;
		const matrix& bias            = *bias_it;
		
		weighted_inputs.emplace_back(multiply(weight, last_activation) + bias);	
		activations.emplace_back(elementwise_apply(*std::prev(std::cend(weighted_inputs)), sigmoid));
	}
	
	// backward pass
	// we start from the final layer and calculate δ^l_j = \pdv{C_x}{z^l_j}
	// using the formula δ^l = (w^{l+1})^T δ^{l+1} \odot σ'(z^l),
	// starting from δ^L = (a^L - y) \odot σ'(z^L) where L is the number of layers
	// then the partial derivatives \pdv{C}{w^l_{kj}} can be found in terms of 
	// δ^l and the other quantities we have already calculated
	
	auto activation_it     = std::crbegin(activations);
	auto weighted_input_it = std::crbegin(weighted_inputs);
	auto weight_it         = std::crbegin(m_weights);

	auto grad_weight_it = std::rbegin(grad.weights);
	auto grad_bias_it   = std::rbegin(grad.biases);

	// start by calculating δ^L 	
	matrix delta = elementwise_multiply(cost_derivative(*activation_it, sample.label),
			elementwise_apply(*weighted_input_it, sigmoid_derivative));

	// note that δ^l and z^l have the same index in the formula , so first advance the (reverse)
	// iterator
	++activation_it;
	
	// \pdv{C}{b^l} = δ^l
	*grad_bias_it = delta;
	// \pdv{C}{w^l_{jk} = a^{l-1}_k δ^l_j
	*grad_weight_it = multiply(
			delta,
			transpose(*activation_it));
	
	++grad_weight_it;
	++grad_bias_it;	
	++activation_it;
	++weighted_input_it;

	// grad.weights and grad.biases have the same number of elements which is
	// network_layer_size.size() - 1, the same number as m_weights, m_biases and weighted_inputs
	//
	// activations has one extra element at the beginning (the sample.image), so total number of
	// elements is network_layer_size.size()
	while(grad_weight_it != std::rend(grad.weights))
	{
		delta = elementwise_multiply(
				multiply(transpose(*weight_it), delta),
				elementwise_apply(*weighted_input_it, sigmoid_derivative));

		*grad_bias_it = delta;
		*grad_weight_it = multiply(delta, transpose(*activation_it));
		
		++grad_weight_it;
		++grad_bias_it;	
		++activation_it;
		++weighted_input_it;
		++weight_it;
	}
	
	return grad;
}

} // namespace thwmakos
