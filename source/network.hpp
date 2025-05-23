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

#include <algorithm>
#include <span>

namespace thwmakos {

// calculates the function 1 / (1 + e^{-x}) 
FloatType sigmoid(FloatType x);

// calculates derivative of sigmoid which is e^{-x} / (1 + e^{-x})^2
FloatType sigmoid_derivative(FloatType x);

// helper function to convert the output of the
// network::evaluate function to an int, representing 
int output_to_int(const column_vector &output);

// concept for a pair of output layer and loss function
template<typename T>
concept loss_function = requires(T t,
								const column_vector &eval,
								const column_vector &expected)
{
	// return loss associated with input
	{ t.loss(eval, expected) } -> std::same_as<FloatType>;
};

// forward declaration from data_loader.hpp
struct training_sample;
class data_loader;

class network
{
	public:
		// constructor, allocates weight and bias matrices according to network_layers
		// and initialises all the weights and biases randomly 
		// from a normal distribution
		explicit network(std::span<const int> network_layers);
		network(const network &) = delete;
		network(network &&)      = default;	
		network &operator=(const network &) = delete;
		network &operator=(network &&)      = default;

		// evaluate the network
		// takes a column vector whose length must match the length of the input layer
		// returns a vector representing the activation of the final layer
		column_vector evaluate(const column_vector &input) const;

		// train the model using mnist data
		// internally uses the data_loader class

		// each using batch_size number of variables, at each step moving
		// by 'learning_rate' towards the direction of steepest descent
		void train(int epochs, int batch_size, FloatType learning_rate, FloatType regularisation_param);

	//private:
	
		// represent gradient of the cost function evaluated at a point with respect
		// to the weights and biases 
		// each entry in the matrices stands for the partial derivative of the cost 
		// functions with respect to the corresponding variable in the m_weights or
		// m_biases matrices
		struct gradient
		{
			// FIXME: should I store a reference to layer structure here?
			std::vector<matrix> weights;
			std::vector<column_vector> biases;
			
			// construct a gradient by properly resizing the
			// weights and biases matrices given as input 
			// the layers of the network
			explicit gradient(std::span<const int> network_layers) : 
				weights(network_layers.size() - 1), 
				biases(network_layers.size() - 1)
			{
				resize_matrices(network_layers);
			}

			gradient(const gradient &) = default;
			gradient(gradient &&) = default;
			gradient &operator=(const gradient &) = default;
			gradient &operator=(gradient &&) = default;
			~gradient() = default;
			
			void resize_matrices(std::span<const int> layers)
			{
				// resize weight matrices in grad
				std::for_each(weights.begin(), weights.end(), 
						[layers_it = layers.begin()] (auto &w) mutable
						{
							const int num_rows = *(layers_it + 1);
							const int num_cols = *layers_it;
							w.set_size(num_rows, num_cols);
							++layers_it;
						});
				// resize bias matrices in grad
				std::for_each(biases.begin(), biases.end(),
						[layers_it = layers.begin()] (auto &b) mutable
						{
							const int num_rows = *(layers_it + 1);
							b.set_size(num_rows);
							++layers_it;
						});
			}
		};

		// stochastic gradient descent
		// minimise cost function based on the arguments	
		void sgd(const data_loader &dl, std::span<const int> sample_indices, FloatType learning_rate, FloatType regularisation_param);

		// calculate partial derivates with respect to 
		// weights and biases for a matrix of training samples, where 
		// each column of the input matrix corresponds to an input
		gradient backpropagation(const matrix &inputs, const matrix &expected_outputs) const; 

		// each element is the number of neurons in the corresponding layer
		// first element should be 28 * 28, last element should be 10
		std::vector<int> m_layers;

		// weight matrices for every layer except first one
		// the weights of a given neuron in layer are represented by a 
		// matrix row, with one entry for every neuron in the previous layer 
		std::vector<matrix> m_weights;
		
		// column vectors for biases
		// first layer neurons do not have biases
		std::vector<column_vector> m_biases;	

};

} // namespace thwmakos

#endif // NETWORK_HPP_INCLUDED
