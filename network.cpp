//
// ~thwmakos~
//
// Tue 18 Jun 02:01:08 BST 2024
//
// network.cpp
//

#include "network.hpp"
#include "data_loader.hpp"
#include "matrix.hpp"

#include <cmath>
#include <random>
#include <algorithm>
#include <span>
#include <format>
#include <print>
#include <chrono>

// used to enable/disable code for debugging
#ifdef NDEBUG 
	constexpr bool g_debug = false;
#else
	constexpr bool g_debug = true;
#endif

namespace thwmakos {

// debug helper
FloatType weight_max(const network::gradient& grad)
{
	if constexpr (g_debug)
	{

		auto wmax = *std::max_element(grad.weights[0].cbegin(), grad.weights[0].cend());

		for(auto i = 1; i < static_cast<int>(grad.weights.size()); ++i)
		{
			auto temp = std::max_element(grad.weights[i].cbegin(), grad.weights[i].cend());

			if(wmax < *temp)
			{
				wmax = *temp;
			}
		}

		return wmax;
	}
	// if not in debug build reduce the function to no op
	else
	{
		return FloatType {};
	}
}


// calculates the function 1 / (1 + e^{-x})
// TODO: add test for this function 
FloatType sigmoid(FloatType x)
{
	// return 0 if x is smaller than this value to
	// avoid NaN's when exp(-x) gets too large
	constexpr FloatType cutoff = -30.0;

	if(x <= cutoff)
	{
		return static_cast<FloatType>(0.0);
	}

	constexpr FloatType one = 1.0;
	return one / (one + std::exp(-x));
}

// calculates derivative of sigmoid
// TODO: add test for this function too
FloatType sigmoid_derivative(FloatType x)
{
	// when to just return 0
	constexpr FloatType cutoff = -25.0;

	if(x <= cutoff)
	{
		return static_cast<FloatType>(0.0);
	}

	constexpr FloatType one = 1.0;
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

void network::train(int epochs, int batch_size, FloatType learning_rate)
{
	constexpr auto images = "../data/train-images-idx3-ubyte";
	constexpr auto labels = "../data/train-labels-idx1-ubyte";

	data_loader dl(images, labels);

	int num_samples = 0;

	if constexpr (g_debug)
	{
		// for debug reduce number of samples
		// for faster execution
		num_samples = 1000;
	}
	else
	{
		// for release builds use all samples
		num_samples = dl.num_samples();
	}

	// prepare a vector indices for the samples
	std::vector<int> indices(num_samples);
	std::generate(indices.begin(), indices.end(), [n = 0] mutable { return n++; });
	
	// need a random engine to shuffle indices
	std::random_device rd {};
	std::default_random_engine eng { rd() };
		
	for(auto epoch = 0; epoch < epochs; ++epoch)
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		std::shuffle(indices.begin(), indices.end(), eng);
		
		// FIXME: do I need to check if batch_size does not divide num_samples -- NO
		for(auto indices_it = indices.cbegin(); indices.cend() - indices_it >= batch_size; indices_it += batch_size)
		{
			// TODO: parallelise this
			stochastic_gradient_descent(dl, std::span<const int> {indices_it, indices_it + batch_size}, learning_rate);
		}
		
		auto t2 = std::chrono::high_resolution_clock::now();
		std::println("Finished epoch {} in {}", epoch, std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1));
	}
}

void network::stochastic_gradient_descent(const data_loader& dl, std::span<const int> sample_indices, FloatType learning_rate)
{
	// in notation: η / <no. of samples per SGD step>
	const auto coeff = learning_rate / static_cast<FloatType>(sample_indices.size());

	// for each sample, we calculate the weight and bias gradients 
	// and store them in this variable
	network::gradient total_gradient;
	// set appropriate dimensions and allocate memory for total_gradient
	for(auto i = 0; i < static_cast<int>(network_layer_size.size() - 1); ++i)
	{
		total_gradient.weights[i].set_size(m_weights[i].size());
		total_gradient.biases[i].set_size(m_biases[i].size());
	}

	for(const int index : sample_indices)
	{
		auto sample = dl.get_sample(index);
		auto grad = backpropagation(sample);
		
		// update total_gradient	
		for(auto i = 0; i < static_cast<int>(network_layer_size.size() - 1); ++i)
		{
			total_gradient.weights[i] += grad.weights[i];
			total_gradient.biases[i]  += grad.biases[i];
		}
	}
	
	// update m_weights and m_biases (stochastic gradient descent step)
	for(auto i = 0; i < static_cast<int>(network_layer_size.size() - 1); ++i)
	{
		m_weights[i] -= coeff * total_gradient.weights[i];
		m_biases[i] -= coeff * total_gradient.biases[i];
	}
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
		
		weighted_inputs.emplace_back(multiply(*weight_it, last_activation) + *bias_it);	
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
