//
//  Matrix.h
//  SparseAutoencoder
//
//  Created by Sam Bodanis on 19/06/2013.
//  Copyright (c) 2013 Sam Bodanis. All rights reserved.
//

#ifndef __SparseAutoencoder__Matrix__
#define __SparseAutoencoder__Matrix__

#include <iostream> 
#include <cmath>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

using namespace boost::numeric::ublas;

class Matrix {
public:
    
    Matrix();
    
    Matrix(int rows, int cols);
        
    ~Matrix();
    
//    Matrix & operator=(const Matrix & rhs) {
//        
//    }
    
    int rows();
    
    int columns();
    
    double get(int i, int j);

    void set(int i, int j, double value);
    
    void equal_size(Matrix & B);
    
    // Returns the identity matrix.
    static Matrix identity(int side_length);
    
    Matrix transpose();

    Matrix multiply(Matrix & B);
    
    Matrix hadamard_product(Matrix & B);
    
    Matrix elementwise_division(Matrix & B);
    
    Matrix elementwise_log();
    
    Matrix elementwise_addition(Matrix & B);
    
    Matrix scalar_addition(double scalar);
    
    Matrix scalar_multiplication(double scalar);

    Matrix scalar_division(double scalar);

    Matrix scalar_minus_matrix(double scalar);
    
    // Euclidean distance between matrices
    double distance(Matrix & B);
    
    // Computes ||A|| = sqrt of sum of squares
    Matrix norm();
    
    // Sticks matrix B to matrix A forming [A B] iff A.size == B.size.
    Matrix append_matrix(Matrix & B);
    
    // Returns the specified column
    Matrix get_column(int index);
    
    // Returns the matrix without the specified column
    Matrix remove_column(int index);
    
    // Attaches a vector of a specified value to the
    // left of the matrix. 
    Matrix prepend_column(int value);
    
    // Changes an MxN matrix 'A' to 1xN vector 'B' where
    // B's column values are the sum of A's values 
    // in that column.
    Matrix sum_columns_to_row_vector();

    bool equals(Matrix & B);
    
    void display();
    
private:

    matrix<double> m;
    
//    double **internal_;

//    int row_count_;

//    int column_count_;
    
    void bounds_check(int i, int j);

    enum elementwise_operation_enum {E_HADAMARD, E_DIVISION, E_ADDITION};
    Matrix elementwise_operation(Matrix & B, 
            elementwise_operation_enum operation_type);
    
    enum scalar_operation_enum {S_ADDITION, S_MULTIPLICATION, S_DIVISION, S_N_MINUS, S_LOG};
    Matrix scalar_operation(double scalar, 
            scalar_operation_enum operation_type);
    
};





#endif /* defined(__SparseAutoencoder__Matrix__) */
