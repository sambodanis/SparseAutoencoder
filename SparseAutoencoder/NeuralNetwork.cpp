//
//  NeuralNetwork.cpp
//  SparseAutoencoder
//
//  Created by Sam Bodanis on 20/06/2013.
//  Copyright (c) 2013 Sam Bodanis. All rights reserved.
//

#include "NeuralNetwork.h"

const double epsilon_init = 0.12;

NeuralNetwork::NeuralNetwork(matrix<double> & training_data,
				  			 matrix<double> & training_labels,
				  			 int max_iterations,
				  			 std::vector<int> hidden_layer_sizes,
				  			 double lambda,
				 			 double alpha,
				 			 double beta,
                             bool sparse) {
	max_iterations_ = max_iterations;
	lambda_ = lambda;
	alpha_ = alpha;
	beta_ = beta;
    sparse_ = sparse;
    hidden_layers_ = (int)hidden_layer_sizes.size();
    for (int i = 0; i < hidden_layers_ + 1; i++) {
        if (i == 0) {
            theta_.push_back(initialize_weights((int)training_data.size2(), hidden_layer_sizes[i]));
        } else if (i == hidden_layer_sizes.size()) {
            theta_.push_back(initialize_weights(hidden_layer_sizes[i - 1], (int)training_labels.size2()));
        } else {
            theta_.push_back(initialize_weights(hidden_layer_sizes[i - 1], hidden_layer_sizes[i]));
        }
    }
	x_data_ = training_data;
	y_data_ = training_labels;
}

NeuralNetwork::~NeuralNetwork() {}

// Runs a feed forward through the network to get a prediction on
// some input data and then test's the similarity between the
// prediction and the input data. 
//double NeuralNetwork::predict(matrix<double> & data_to_predict) {
//    prepend_column(data_to_predict, 1);
//    matrix<double> guess (data_to_predict.size1(), data_to_predict.size2());
//    for (int i = 0; i < theta_.size(); i++) {
//        guess = prod(prepend_column(data_to_predict, 1), trans(theta_[i]));
//        guess = sigmoid(guess);
//    }
//    return 1;
//}

void NeuralNetwork::train() {
    gradient_descent();
//    predict(x_data_);
}

//// Makes a prediction on a set of points
//// Measures how far off a guess is
//// Computes: J = sum(1/m * sum(-y.*log(a3) - (1 - y).*log(1-a3)))
//double NeuralNetwork::compute_cost_logistic() {
//    double m = x_data_.rows();
//    feed_forward();
//    matrix<double> a3 = feed_forward_map_["a3"];
//    matrix<double> negative_y = y_data_.scalar_multiplication(-1.0); // -y
//    matrix<double> log_a3 = a3.elementwise_log(); // log(a3)
//    matrix<double> negative_y_log_a3 = negative_y.hadamard_product(log_a3); // -y.*log(a3)
//    matrix<double> one_minus_y = y_data_.scalar_minus_matrix(1); // 1 - y
//    matrix<double> one_minus_a3 = a3.scalar_minus_matrix(1); // 1 - a3
//    matrix<double> log_one_minus_a3 = one_minus_a3.elementwise_log(); // log(1 - a3)
//    matrix<double> one_minus_y_log_a3 = one_minus_y.hadamard_product(log_one_minus_a3); // (1 - y).*log(1 - a3)
//    matrix<double> negative_one_minus_y_log_a3 = one_minus_y_log_a3.scalar_multiplication(-1); // -(1 - y).*log(1 - a3)
//    matrix<double> first_internal_sum = negative_one_minus_y_log_a3
//        .elementwise_addition(negative_one_minus_y_log_a3); // -y.*log(a3) -(1 - y).*log(1 - a3)
//    matrix<double> first_sum = first_internal_sum.sum_columns_to_row_vector(); // sum(-newY.*log(a3) - (1 - newY).*log(1-a3))
//    matrix<double> second_internal_sum = first_sum.scalar_multiplication(1.0 / m); // 1/m * sum(-newY.*log(a3) - (1 - newY).*log(1-a3))
//    matrix<double> second_sum = second_internal_sum.transpose().sum_columns_to_row_vector(); // Might be an issue, sum(1/m * sum(-newY.*log(a3) - (1 - newY).*log(1-a3)))
//    return second_sum.get(0,0) + regularize_cost();
//}

// Measures how far off a guess is
// Computes: J = sum(1/m * sum(||a3 - y||^2))
double NeuralNetwork::compute_cost_squared_error() {
    double m = x_data_.size1();
    feed_forward();
    matrix<double> a3 = neuron_activations_.back();//[neuron_activations_.size() - 1];
//    std::cout << a3.size1() << "," << a3.size2() << " -a3,y-   " << y_data_.size1() << "," << y_data_.size2() << std::endl;
    
    matrix<double> norm_a3_minus_y = distance(a3, y_data_); // ||a3 - y|| TODO check this is the correct norm
    matrix<double> norm_a3_minus_y_squared = element_prod(norm_a3_minus_y, norm_a3_minus_y); // ||a3 - y||^2
    
    matrix<double> first_sum = sum_columns_to_row_vector(norm_a3_minus_y_squared); // sum(||a3 - y||^2)
    first_sum *= 1.0 / m; // 1 / m * sum(||a3 - y||^2)
    matrix<double> first_sum_transpose = trans(first_sum);
    matrix<double> second_sum = sum_columns_to_row_vector(first_sum_transpose); // Might be an issue, sum(1 / m * sum(||a3 - y||^2))
    double cost = second_sum(0, 0) - regularize_cost();
    return sparse_ ? cost + kullback_leibler_divergence() : cost;

}

double NeuralNetwork::kullback_leibler_divergence() {
    double result = 0;
    for (int j = 0; j < hidden_unit_average_activation_.size2(); j++) {
        double left = sparsity_parameter_ * log(sparsity_parameter_ / hidden_unit_average_activation_(0, j));
        double right = (1 - sparsity_parameter_) * log((1 - sparsity_parameter_) / (1 - hidden_unit_average_activation_(0, j)));
        result += left + right;
    }
    return beta_ * result;
}

// Adds regularisation to suppress the impact of too many higher order terms.
// ie smooths out a 7th order polynomial fitted to 8 points.
// This enables the network to generalise better.
// Computes: J = J + (lambda / (2 * m)) * sum(sum(sum(Theta1(:, 2:end).^2)) +
//                                            sum(sum(Theta2(:, 2:end).^2)))
// The last line get(0,0)'s are done because by that point the MxN matrix<double> has
// been summed to a 1xN vector which is then transposed to a Nx1 matrix<double> which is
// is then summed to a 1x1 scalar value at the 0,0 index of the matrix<double> object.
double NeuralNetwork::regularize_cost() {
    double total = 0;
    for (int i = 0; i < theta_.size(); i++) {
        matrix<double> theta_squared = element_prod(theta_[i], theta_[i]);
        matrix<double> theta_sq_transpose = trans(sum_columns_to_row_vector(theta_squared));
        matrix<double> full_sum = sum_columns_to_row_vector(theta_sq_transpose);
        total += full_sum(0, 0);
    }
    return (lambda_ / (2.0 * x_data_.size1())) * total;
}


// Computes a guess of which label a set of points belong to.
// Columns of 1 are prepended to take into account bias.
// the 'a' matrices hold the activations of the hidden units
// in a layer and the 'z' matrices are the total weighted
// sum of the activations of each unit in the layer to be
// fed into the next layer.
// a1 isn't put through the sigmoid funciton because it is
// the input data.
// Output doesn't have bias units added. 
// activations = ax
// weights = zx
void NeuralNetwork::feed_forward() {
    for (int i = 0; i < theta_.size() + 1; i++) {
        if (i == 0) {
            if (neuron_activations_.size() < theta_.size() + 1) {
                neuron_activations_.push_back(prepend_column(x_data_, 1));
                matrix<double> holder (0, 0);
                neuron_weights_.push_back(holder);
            } else {
                neuron_activations_[i] = prepend_column(x_data_, 1);
            }
        } else {
            if (neuron_activations_.size() < theta_.size() + 1) {
                neuron_weights_.push_back(prod(neuron_activations_[i - 1], trans(theta_[i - 1])));
                matrix<double> sigmoid_activation = sigmoid(neuron_weights_[i]);
                if (i == theta_.size()) neuron_activations_.push_back(sigmoid_activation);
                else neuron_activations_.push_back(prepend_column(sigmoid_activation, 1));
            } else {
                neuron_weights_[i] = prod(neuron_activations_[i - 1], trans(theta_[i - 1]));
                matrix<double> sigmoid_activation = sigmoid(neuron_weights_[i]);
                if (i == theta_.size()) neuron_activations_[i] = sigmoid_activation;
                else neuron_activations_[i] = prepend_column(sigmoid_activation, 1);
            }
        }
        if (sparse_ && i == hidden_layers_) {
            hidden_unit_average_activation_ = sum_columns_to_row_vector(neuron_activations_[i]);
            // TODO: might have to add the x^i thing in here.
        }
    }
}

// Computes the derivatives of the units in each layer.
// Derivatives are computed recursivly using the delta rule
// outer layer delta = -(y - a^nl).*f'(z^nl)
// innter layer deltas = ((W^l)'delta^(l+1)).*f'(z^l)
// theta_gradients are then set with:
//      grad_J = delta^(l+1) * (a^l)'
// Note, gradient matrix array is reversed at the end
// of the function so that matrix 0 corresponds to index 0. 
void NeuralNetwork::compute_gradients() {
    feed_forward();
    std::vector<matrix<double> > deltas;
    for (int i = (int)theta_.size(); i > 0; i--) {
        if (i == theta_.size()) {
            matrix<double> a3 = neuron_activations_.back();
//            print_matrix_size(y_data_, "y_data");
            a3 -= y_data_;
            a3 = element_prod(a3, sigmoid_gradient(neuron_weights_.back()));
            a3 *= -1;
            deltas.push_back(a3);
        } else {
            matrix<double> previous_delta = trans(deltas.back());
            matrix<double> transpose_weights = trans(theta_[i]);            
            matrix<double> sigmoid_weights = sigmoid_gradient(neuron_weights_[i]);
            matrix<double> weights_times_delta = trans(prod(transpose_weights, previous_delta));
//            if (sparse_ && i == hidden_layers_) {
//                matrix<double> left_temp(hidden_unit_average_activation_.size1(), hidden_unit_average_activation_.size2(), -1 * sparsity_parameter_);
//                matrix<double> left = element_div(left_temp, hidden_unit_average_activation_);
//                matrix<double> right_temp(hidden_unit_average_activation_.size1(), hidden_unit_average_activation_.size2(), 1 - sparsity_parameter_);
//                matrix<double> right = element_div(right_temp, scalar_minus_matrix(hidden_unit_average_activation_, 1));
//                matrix<double> sparse_kl_divergence = beta_ * (left + right);
//                for (int i = 0; i < weights_times_delta.size1(); i++) {
//                    for (int j = 0; j < weights_times_delta.size2(); j++) {
//                        weights_times_delta(i, j) -= sparse_kl_divergence(0, j);
//                    }
//                }
//            }
            matrix<double> new_delta = element_prod(remove_column(weights_times_delta, 0), sigmoid_weights);
            deltas.push_back(new_delta);
        }
    }
//    debug_matrices(deltas);
    int activation_iterator = (int)deltas.size() - 1;
    for (int i = 0; i < deltas.size(); i++) {
        // TODO: seperate core and bias, operate then reconstruct. 
        matrix<double> delta = deltas[i];
        matrix<double> activations = neuron_activations_[activation_iterator--];
        matrix<double> delta_activation = prod(trans(delta), activations);
        matrix<double> gradient = delta_activation;
        if (theta_grad_.size() < theta_.size()) theta_grad_.push_back(gradient);
        else theta_grad_[i] = gradient;
    }
    std::reverse(theta_grad_.begin(), theta_grad_.end());
    regularize_gradients();
}

// Adds regularisation to gradient matrices.
void NeuralNetwork::regularize_gradients() {
//    for (int i = 0; i < theta_grad_.size(); i++) {
//        std::string name = "theta_grad";
//        std::cout << i + 1 << "  ";
//        print_matrix_size(theta_grad_[i], name);
//    }
    for (int i = 0; i < theta_.size(); i++) {
        matrix<double> theta_1st_column = get_column(theta_grad_[i], 0);
        matrix<double> theta_other_columns = remove_column(theta_grad_[i], 0);
        matrix<double> regularize_theta = remove_column(theta_[i], 0);
        regularize_theta *= (lambda_ / x_data_.size2());
        theta_other_columns += regularize_theta;
        theta_grad_[i] = append_matrix(theta_1st_column, theta_other_columns);
    }
}

// Computes the activations of every unit in a matrix<double>
// using the hyperbolic tangent function.
matrix<double> NeuralNetwork::sigmoid(matrix<double> & mat) {
    matrix<double> result((int)mat.size1(), (int)mat.size2());
    for (int i = 0; i < mat.size1(); i++) {
        for (int j = 0; j < mat.size2(); j++) {
            result(i, j) = std::tanh(mat(i, j));
        }
    }
    return result;
}

// Derivative of tanh(z) is 1 - tanh(z)^2
matrix<double> NeuralNetwork::sigmoid_gradient(matrix<double> & mat) {
    matrix<double> sigmoid_result = sigmoid(mat);
    matrix<double> all_ones(mat.size1(), mat.size2(), 1);
    all_ones -= element_prod(sigmoid_result, sigmoid_result);
    return all_ones;
}

// TODO figure out a better way of generating random numbers.
matrix<double> NeuralNetwork::initialize_weights(int layers_in, int layers_out) {
    matrix<double> result(layers_out, layers_in + 1);

	for (int i = 0; i < result.size1(); i++) {
		for (int j = 0; j < result.size2(); j++) {
            result(i, j) = double_rand(0, 1) * 2 * epsilon_init - epsilon_init;
		}
	}
    return result;
}

// TODO update this to a more advanced optimizer eg conjugate gradient. 
// Runs gradient descent.
// A guess is made by computing a feed forward using the theta matrices.
// The theta matrix<double> weights are then nudged in the correct direction by
// iteratively computing gradients of every element and taking a small step in
// the direction which minimises the error.
void NeuralNetwork::gradient_descent() {
    int iteration = 0;
    int row = arc4random();
    int col = arc4random();
    double cost = compute_cost_squared_error();
    while (iteration++ < max_iterations_) {
        compute_gradients();
        cost = compute_cost_squared_error();
        for (int i = 0; i < theta_.size(); i++) {
            theta_[i] = theta_[i] - alpha_ * theta_grad_[i];
        }
        row = row % theta_[0].size1();
        col = col % theta_[0].size2();
//        std::cout << "Theta_1(" << row << "," << col << ") = " << theta_[0](row, col) << std::endl;
        std::cout << "Iteration   " << iteration << " | Cost: " << cost << std::endl;
    }
}

matrix<double> NeuralNetwork::scalar_minus_matrix(matrix<double> & mat, double scalar) {
    matrix<double> result((int)mat.size1(), (int)mat.size2());
    for (int i = 0; i < result.size1(); i++) {
        for (int j = 0; j < result.size2(); j++) {
            result(i, j) = scalar - mat(i, j);
        }
    }
    return result;

}

matrix<double> NeuralNetwork::prepend_column(matrix<double> & mat, double value) {
    matrix<double> result((int)mat.size1(), (int)mat.size2() + 1);
    for (int i = 0; i < result.size1(); i++) {
        for (int j = 0; j < result.size2(); j++) {
            if (j == 0) {
                result(i, j) = value;
            } else {
                result(i, j) = mat(i, j-1);
            }
        }
    }
    return result;
}

matrix<double> NeuralNetwork::distance(matrix<double> & mat_1, matrix<double> & mat_2) {
    matrix<double> result (mat_1.size1(), mat_1.size2());
    for (int i = 0; i < mat_1.size1(); i++) {
        for (int j = 0; j < mat_1.size2(); j++) {
            result(i, j) = pow(mat_1(i, j) - mat_2(i, j), 2);
        }
    }
    return result;
}

// Changes an MxN matrix 'A' to 1xN vector 'B' where
// B's column values are the sum of A's values
// in that column.
matrix<double> NeuralNetwork::sum_columns_to_row_vector(matrix<double> & mat) {
    matrix<double> result (1, mat.size2());
    for (int i = 0; i < result.size1(); i++) {
        for (int j = 0; j < result.size2(); j++) {
            result(0, j) = result(0, j) + mat(i, j);
        }
    }
    return result;
}

matrix<double> NeuralNetwork::get_column(matrix<double> & mat, int index) {
    matrix<double> result (mat.size1(), 1);
    for (int i = 0; i < result.size1(); i++) {
        result(i, 0) = mat(i, index);
    }
    return result;
}

matrix<double> NeuralNetwork::remove_column(matrix<double> & mat, int index) {
    matrix<double> result (mat.size1(), mat.size2() - 1);
    for (int i = 0; i < result.size1(); i++) {
        for (int j = 0; j < result.size2(); j++) {
            if (j < index) {
                result(i, j) = mat(i, j);
            } else {
                result(i, j) = mat(i, j + 1);
            }
        }
    }
    return result;
}

matrix<double> NeuralNetwork::append_matrix(matrix<double> & mat_1, matrix<double> & mat_2) {
    matrix<double> result (mat_1.size1(), mat_1.size2() + mat_2.size2());
    for (int i = 0; i < result.size1(); i++) {
        for (int j = 0; j < result.size2(); j++) {
            if (j < mat_1.size2()) {
                result(i, j) = mat_1(i, j);
            } else {
                result(i, j) = mat_2(i, j - mat_1.size2());
            }
        }
    }
    return result;
}

void NeuralNetwork::print_matrix_size(matrix<double> & mat, std::string name) {
    std::cout << "Size of " << name << ": " << mat.size1() << "," << mat.size2() << std::endl;
}

void NeuralNetwork::debug_matrices(std::vector<matrix<double> > & deltas) {
    for (int i = 0; i < deltas.size(); i++) {
        std::string name = "delta";
        std::cout << i + 1 << "  ";
        print_matrix_size(deltas[i], name);
    }
    for (int i = 0; i < theta_.size(); i++) {
        std::string name = "theta";
        std::cout << i + 1 << "  ";
        print_matrix_size(theta_[i], name);
    }
    for (int i = 0; i < neuron_activations_.size(); i++) {
        std::string name = "a";
        std::cout << i + 1 << "  ";
        print_matrix_size(neuron_activations_[i], name);
    }
    for (int i = 0; i < neuron_weights_.size(); i++) {
        std::string name = "z";
        std::cout << i + 1 << "  ";
        print_matrix_size(neuron_weights_[i], name);
    }
}

double NeuralNetwork::double_rand(double min, double max) {
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
}















