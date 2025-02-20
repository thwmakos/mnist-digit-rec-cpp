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
#include <execution>
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

int output_to_int(const column_vector &output)
{
	return std::distance(output.cbegin(), std::max_element(output.cbegin(), output.cend()));
}

// returns a column vector with the partial derivatives of the cost function
// in notation this returns \pdv{C_x}{a^L} for the sample (x, y)
// where a^L is output_activations and y is label 
matrix cost_derivative(const matrix &output_activations, const matrix &label)
{
	return output_activations - label;
}

network::network(std::span<const int> network_layers) : 
	m_layers(network_layers.begin(), network_layers.end()),
	m_weights(m_layers.size() - 1),
	m_biases(m_layers.size() - 1)
{
	if(*network_layers.begin() != 28 * 28 || *(network_layers.end() - 1) != 10)
	{
		throw std::invalid_argument(std::format(
			"Input layer should have 28 * 28 neurons, output should have 10, found {} and {}",
			*network_layers.begin(), *(network_layers.end() - 1)));
	}


	// allocate space for the weight and bias matrices
	//m_weights[0].set_size(network_layer_size[1], network_layer_size[0]);
	//m_weights[1].set_size(network_layer_size[2], network_layer_size[1]);
	//m_biases[0].set_size(network_layer_size[1], 1);
	//m_biases[1].set_size(network_layer_size[2], 1);
	// resize weight matrices in grad
	
	std::for_each(m_weights.begin(), m_weights.end(), 
			[layers_it = m_layers.begin()] (auto &w) mutable
			{
				const int num_rows = *(layers_it + 1);
				const int num_cols = *layers_it;
				w.set_size(num_rows, num_cols);
				++layers_it;
			});
	// resize bias matrices in grad
	std::for_each(m_biases.begin(), m_biases.end(),
			[layers_it = m_layers.begin()] (auto &b) mutable
			{
				const int num_rows = *(layers_it + 1);
				b.set_size(num_rows);
				++layers_it;
			});
	
	// initialise matrices with random normally distributed entries
	std::random_device rd {}; 
	std::default_random_engine eng { rd() };
	
	// lambda to randomise an sequence of matrices using 
	// the random distribution constructed above
	auto randomise = [&eng] (auto &matrices)
	{
		for(auto &&mat : matrices)
		{
			const auto [num_rows, num_cols] = mat.size();
		
			// num_cols gives the amount of input neurons in 
			// this layer, thus initialise the weights from a 
			// N(0, 1 / sqrt(number of input neurons)) 	
			const float stddev = 1.0f / std::sqrtf(num_cols);
				
			std::normal_distribution<FloatType> normal(0.0f, stddev);

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

column_vector network::evaluate(const column_vector &input) const
{
	// check if input size is correct
	// has to be equal to the number of input layers
	if(input.num_rows() != m_weights[0].num_cols())	
	{
		throw std::invalid_argument("expected column vector with length of input layer"); 
	}

	//matrix result {};	
	//result = elementwise_apply(m_weights[0] * input  - m_biases[0], sigmoid);
	//result = elementwise_apply(m_weights[1] * result - m_biases[1], sigmoid);

	auto result = input;	

	for(int i = 0; i < static_cast<int>(m_weights.size()); ++i)
	{
		result = elementwise_apply(m_weights[i] * result - m_biases[i], sigmoid);
	}

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
		num_samples = 100;
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
	
	std::println("Started training for {} epochs, with batch size of {} and learning rate {}", epochs, batch_size, learning_rate);	

	for(auto epoch = 0; epoch < epochs; ++epoch)
	{
		auto t1 = std::chrono::high_resolution_clock::now();
		std::shuffle(indices.begin(), indices.end(), eng);
		
		// FIXME: do I need to check if batch_size does not divide num_samples -- NO
		for(auto indices_it = indices.cbegin(); indices.cend() - indices_it >= batch_size; indices_it += batch_size)
		{
			sgd(dl, std::span<const int> {indices_it, indices_it + batch_size}, learning_rate);
		}
		
		auto t2 = std::chrono::high_resolution_clock::now();
		std::println("Finished epoch {} in {}", epoch + 1, std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1));
	}
}


network::gradient network::backpropagation(const matrix &inputs, const matrix &expected_outputs) const
{
	const int num_samples = inputs.num_cols();
	const int num_layers  = static_cast<int>(m_layers.size());

	network::gradient total_gradient(std::span(m_layers.cbegin(), m_layers.cend()));
	std::vector<network::gradient> gradients(num_samples, total_gradient); // one gradient for each
																		   // sample (i.e. each
																		   // column)

	std::vector<matrix> activations { inputs }; // first activation is the input (training samples)
	std::vector<matrix> weighted_inputs {};
	
	// forward pass

	for(int i = 0; i < num_layers - 1; ++i)
	{
		// z^l = w^l * a^{l - 1} + b^l
		// z^l is a matrix with each column representing the activation of each training sample
		auto weighted_input = add_column(m_weights[i] * activations.back(), m_biases[i]);
		
		activations.push_back(elementwise_apply(weighted_input, sigmoid));
		weighted_inputs.push_back(std::move(weighted_input));
	}

	// backward pass
	// we start from the final layer and calculate δ^l_j = \pdv{C_x}{z^l_j}
	// using the formula δ^l = (w^{l+1})^T δ^{l+1} \odot σ'(z^l),
	// starting from δ^L = (a^L - y) \odot σ'(z^L) where L is the number of layers
	// then the partial derivatives \pdv{C}{w^l_{kj}} can be found in terms of 
	// δ^l and the other quantities we have already calculated
	
	// process last layer
	// start by calculating δ^L 	
	auto delta = elementwise_multiply(cost_derivative(activations[num_layers - 1], expected_outputs),
			elementwise_apply(weighted_inputs[num_layers - 2], sigmoid_derivative));
	
	// \pdv{C}{b^l} = δ^l
	// \pdv{C}{w^l_{jk} = a^{l-1}_k δ^l_j
	for(int j = 0; j < num_samples; ++j)
	{
		auto delta_column = get_column(delta, j);
		// using the reshape member here and below to avoid extra allocation
		total_gradient.weights[num_layers - 2] += (1.0 / num_samples) * delta_column * get_column(activations[num_layers - 2], j).to_row(); 	
		total_gradient.biases[num_layers - 2]  += (1.0 / num_samples) * delta_column;
	}
	
	// rest of the layers
	for(int i = num_layers - 3; i >= 0; --i)
	{
		delta = elementwise_multiply(
				transpose(m_weights[i + 1]) * delta,
				elementwise_apply(weighted_inputs[i], sigmoid_derivative));
		
		for(int j = 0; j < num_samples; ++j)
		{
			// \pdv{C}{b^l} = δ^l
			// \pdv{C}{w^l_{jk} = a^{l-1}_k δ^l_j
			auto delta_column = get_column(delta, j);
			total_gradient.weights[i] += (1.0 / num_samples) * delta_column * get_column(activations[i], j).to_row();
			total_gradient.biases[i]  += (1.0 / num_samples) * delta_column;
		}
	}
	
	return total_gradient;
}

void network::sgd(const data_loader &dl, std::span<const int> sample_indices, FloatType learning_rate)
{
	const int num_samples = static_cast<int>(sample_indices.size());

	// load samples into a matrix, one column per sample
	matrix inputs(m_layers[0], num_samples);
	// expected output for each input
	matrix expected_outputs(m_layers.back(), num_samples);

	std::vector<training_sample> samples(num_samples);
	std::transform(sample_indices.begin(), sample_indices.end(), samples.begin(),
			[&dl] (auto sample_index)
			{
				return dl.get_sample(sample_index);
			});

	for(int j = 0; j < inputs.num_cols(); ++j)
	{
		for(int i = 0; i < inputs.num_rows(); ++i)
		{
			inputs[i, j] = samples[j].image[i, 0];
		}
	}
	
	for(int j = 0; j < expected_outputs.num_cols(); ++j)
	{
		for(int i = 0; i < expected_outputs.num_rows(); ++i)
		{
			expected_outputs[i, j] = samples[j].label[i, 0];
		}
	}

	auto total_gradient = backpropagation(inputs, expected_outputs);

	// update m_weights and m_biases (stochastic gradient descent step)
	for(int i = 0; i < static_cast<int>(m_layers.size() - 1); ++i)
	{
		m_weights[i] -= learning_rate * total_gradient.weights[i];
		m_biases[i]  -= learning_rate * total_gradient.biases[i];
	}
}

} // namespace thwmakos
