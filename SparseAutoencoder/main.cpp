//
//  main.cpp
//  SparseAutoencoder
//
//  Created by Sam Bodanis on 19/06/2013.
//  Copyright (c) 2013 Sam Bodanis. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <utility>
#include "NeuralNetwork.h"
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>


void test_numbers();
void train_with_images(std::vector<matrix<double> > & images);
matrix<double> get_patch(matrix<double> & mat, int r_start, int c_start, int rows, int columns);
int rand_in_range(int min, int max);
void test_predict(boost::numeric::ublas::matrix<double> & test_image);
boost::numeric::ublas::matrix<double> load_image(std::string);
boost::numeric::ublas::matrix<double> vec_to_mat(std::pair<int, int> size, std::vector<double> pixel_values);


int main(int argc, const char * argv[]) {
    
//    std::string test_filename = "grayscale.txt";
//    boost::numeric::ublas::matrix<double> test_image = load_image(test_filename);
//    test_predict(test_image);
    test_numbers();
    return 0;
}

void test_numbers() {
    int imgSize = 512;
    int numImages = 10;
    std::string filename = "olsh.dat";
    
    std::vector<boost::numeric::ublas::matrix<double> > images;
    
    std::ifstream indata;
    double num; // variable for input value
    
    indata.open(filename.c_str());
    if(!indata) {
        std::cerr << "Error: file could not be opened" << std::endl;
        return;
    }
    
    for(int i = 0; i < numImages; ++i) {
        boost::numeric::ublas::matrix<double> m(imgSize,imgSize);
        for(int r = 0; r < imgSize; ++r) {
            for(int c = 0; c < imgSize; ++c) {
                if(indata.eof()) {
                    std::cerr << "Error: ran out of input values on (" << r << "," << c << ")" << std::endl;
                    return;
                }
                
                indata >> num;
                m(r,c) = num;
            }
        }
        
        images.push_back(m);
    }
    indata.close();
    
    std::cout << "Input data loaded" << std::endl;
    train_with_images(images);
    return;
}

void train_with_images(std::vector<matrix<double> > & images) {
    int k_max = 10;
    matrix<double> all_training_examples (k_max, 8 * 8);
    for (int k = 0; k < k_max; k++) {
        matrix<double> example = images[(int)arc4random() % images.size()];
        int row_start = rand_in_range(0, (int)example.size1() - 8);
        int col_start = rand_in_range(0, (int)example.size2() - 8);
        matrix<double> image_patch = get_patch(example, row_start, col_start, 8, 8);
        for (int j = 0; j < image_patch.size2(); j++) {
            all_training_examples(k, j) = image_patch(0, j);
        }
    }
    std::vector<int> hidden_layer_sizes;
    hidden_layer_sizes.push_back(30);
    NeuralNetwork net = NeuralNetwork(all_training_examples, all_training_examples, 100, hidden_layer_sizes, 0.1, 0.1, 1, true);
    net.train();
    
    
}

int rand_in_range(int min, int max) {
    return min + (rand() % (int)(max - min + 1));
}

matrix<double> get_patch(matrix<double> & mat, int r_start, int c_start, int rows, int columns) {
    
    matrix<double> result (1, rows * columns);
    int result_index = 0;
    for (int i = r_start; i < r_start + rows; i++) {
        for (int j = c_start; j < c_start + columns; j++) {
            result(0, result_index++) = mat(i, j);
        }
    }
    return result;
}

void test_predict(boost::numeric::ublas::matrix<double> & test_image) {
    test_image.resize(50, 100);
//    std::cout << test_image << std::endl;
    std::vector<int> hidden_layer_sizes;
    hidden_layer_sizes.push_back(25);
    NeuralNetwork net = NeuralNetwork(test_image, test_image, 100, hidden_layer_sizes, 1, 3.3, 0, true);
    net.train();
}

boost::numeric::ublas::matrix<double> load_image(std::string filename) {
    std::ifstream image_file(filename.c_str());
    std::string line = "";
    std::vector<double> pixel_values;
    bool first = true;
    // std::string size;
    std::pair <int, int> size;
    if (image_file.is_open()) {
        while (image_file.good()) {
            getline(image_file, line);
            if (first) {
                first = false;
                std::string second_line;
                getline(image_file, second_line);
                size = std::make_pair(atoi(line.c_str()), atoi(second_line.c_str()));
            } else {
                pixel_values.push_back(atof(line.c_str()));
            }
        }
        image_file.close();
    } else {
        std::cout << "Could not open file" << std::endl;
    }
    return vec_to_mat(size, pixel_values);
}

boost::numeric::ublas::matrix<double> vec_to_mat(std::pair<int, int> size, std::vector<double> pixel_values) {
    boost::numeric::ublas::matrix<double> result(size.second, size.first);
    int k = 0;
    for (int i = 0; i < result.size1(); i++) {
        for (int j = 0; j < result.size2(); j++) {
            result(i, j) = pixel_values[k++];
        }
    }
    return result;
}






