//
//  NeuralNetwork.h
//  SparseAutoencoder
//
//  Created by Sam Bodanis on 20/06/2013.
//  Copyright (c) 2013 Sam Bodanis. All rights reserved.
//

#ifndef __SparseAutoencoder__NeuralNetwork__
#define __SparseAutoencoder__NeuralNetwork__

#include <iostream>
#include <map>
#include <vector>
//#include "Matrix.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>

using namespace boost::numeric::ublas;

class NeuralNetwork {
public:

	NeuralNetwork(matrix<double> & training_data,
				  matrix<double> & training_labels,
				  int max_iterations,
				  std::vector<int> hidden_layer_sizes,
				  double lambda,
				  double alpha,
				  double beta,
                  bool sparse);

	~NeuralNetwork();

//    double predict(matrix<double> & data_to_predict);

    void train();
    
private:
    
//    double compute_cost_logistic();
    
    double compute_cost_squared_error();
    
	double regularize_cost();
    
    double kullback_leibler_divergence();
    
	void feed_forward();
    
	void compute_gradients();
    
	void regularize_gradients();
    
	matrix<double> sigmoid(matrix<double> & mat);
    
	matrix<double> sigmoid_gradient(matrix<double> & mat);
    
	matrix<double> initialize_weights(int layers_in, int layers_out);
    
	void gradient_descent();
    
    matrix<double> scalar_minus_matrix(matrix<double> & mat, double scalar);
    
    matrix<double> prepend_column(matrix<double> & mat, double value);
    
    matrix<double> distance(matrix<double> & mat_1, matrix<double> & mat_2);
    
    matrix<double> sum_columns_to_row_vector(matrix<double> & mat);
    
    matrix<double> get_column(matrix<double> & mat, int index);
    
    matrix<double> remove_column(matrix<double> & mat, int index);
    
    matrix<double> append_matrix(matrix<double> & mat_1, matrix<double> & mat_2);
    
    void print_matrix_size(matrix<double> & mat, std::string name);

    void debug_matrices(std::vector<matrix<double> > & deltas);

    double double_rand(double min, double max);
    
    std::vector<matrix<double> > theta_;
    std::vector<matrix<double> > theta_grad_;
    std::vector<matrix<double> > neuron_activations_;
    std::vector<matrix<double> > neuron_weights_;
	matrix<double> x_data_;
	matrix<double> y_data_;
    double lambda_;
	double alpha_;
	double beta_;
    bool sparse_;
	int max_iterations_;
    double sparsity_parameter_;
    int hidden_layers_;
    matrix<double> hidden_unit_average_activation_;
};

#endif /* defined(__SparseAutoencoder__NeuralNetwork__) */
