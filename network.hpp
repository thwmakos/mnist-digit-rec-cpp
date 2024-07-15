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
#include <span>

namespace thwmakos {

// calculates the function 1 / (1 + e^{-x}) 
FloatType sigmoid(FloatType x);

// calculates derivative of sigmoid which is e^{-x} / (1 + e^{-x})^2
FloatType sigmoid_derivative(FloatType x);

// forward declaration from data_loader.hpp
struct training_sample;
class data_loader;

// neural network layers determined at compile time
// input and output layers are first and last elements
// of the array, so they are included here
constexpr std::array<FloatType, 3> network_layer_size = {28 * 28, 30, 10};

class network
{
	public:
		// default constructor, allocates matrices and initialises 
		// all the weights and biases randomly from a normal distribution
		explicit network();

		// evaluate the network
		// takes a column vector whose length must match the length of the input layer
		// returns a vector representing the activation of the final layer
		matrix evaluate(const matrix& input) const;

		// train the model using mnist data
		// internally uses the data_loader class
		// performs 'epochs' number of stochastic gradient descent steps
		// each using batch_size number of variables, at each step moving
		// by 'learning_rate' towards the direction of steepest descent
		void train(int epochs, int batch_size, FloatType learning_rate);

	//private:
	
		// represent gradient of the cost function evaluated at a point with respect
		// to the weights and biases 
		// each entry in the matrices stands for the partial derivative of the cost 
		// functions with respect to the corresponding variable in the m_weights or
		// m_biases matrices
		//
		// NOTE: the members are vectors here, although they could have been 
		// std::array, but I am trying no be ready for runtime instead of compile-time
		// defined number of layers and neurons
		struct gradient
		{
			std::vector<matrix> weights;
			std::vector<matrix> biases;

			gradient() : weights(network_layer_size.size() - 1), 
							biases(network_layer_size.size() - 1)
			{}

			gradient(const gradient&) = default;
			gradient(gradient &&) = default;
			gradient& operator=(const gradient&) = default;
			gradient& operator=(gradient &&) = default;
			~gradient() = default;
		};

		// minimise cost function based on the arguments	
		void stochastic_gradient_descent(const data_loader& dl, std::span<const int> sample_indices, FloatType learning_rate);

		// adjust weights and biases for a single training sample
		gradient backpropagation(const training_sample&) const;

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
