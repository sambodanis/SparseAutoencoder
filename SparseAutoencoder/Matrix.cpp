//
//  Matrix.cpp
//  SparseAutoencoder
//
//  Created by Sam Bodanis on 19/06/2013.
//  Copyright (c) 2013 Sam Bodanis. All rights reserved.
//

#include "Matrix.h"


Matrix::Matrix() {}

Matrix::Matrix(int rows, int columns) {
    m = matrix<double>(rows, columns);
    
//    internal_ = new double*[rows];
//    for (int i = 0; i < rows; i++) {
//        internal_[i] = new double[columns];
//    }
//    row_count_ = rows;
//    column_count_ = columns;
}

Matrix::~Matrix() {
//    for (int i = 0; i < row_count_; i++) {
//        delete[] internal_[i];
//    }
//    delete internal_;
}

int Matrix::rows() {
    return (int)m.size1();
//    return row_count_;
}

int Matrix::columns() {
    return (int)m.size2();
//    return column_count_;
}

void Matrix::set(int i, int j, double value) {
//    bounds_check(i, j);
    m(i, j) = value;
//    internal_[i][j] = value;
    return;
}

double Matrix::get(int i, int j) {
    return m(i, j);
//    bounds_check(i, j);
//    return internal_[i][j];
}

void Matrix::bounds_check(int i, int j) {
    if (i >= 0 && i < rows() && j >= 0 && j < columns()) {
        return;
    }
    fprintf(stderr, "Index %d,%d out of bounds of Matrix %d,%d", i, j, rows(), 
            columns());
    // int crash = 1/0;
    exit(1);
}

void Matrix::equal_size(Matrix & B) {
    if (rows() != B.rows() || columns() != B.columns()) {
        fprintf(stderr, "Incompatible Matrix sizes: %d,%d * %d,%d", rows(), columns(),
            B.rows(), B.columns());
        // int crash = 1/0;
        exit(1);
    }
    return;
}

Matrix Matrix::transpose() {
    Matrix result = Matrix(columns(), rows());
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < columns(); j++) {
            result.set(i, j, get(j, i));
        }
    }
    return result;
}

Matrix Matrix::multiply(Matrix & B) {
    equal_size(B);
    prod
    return prod(m, B);
    
//    Matrix result = Matrix(rows(), B.columns());
//    for (int i = 0; i < result.rows(); i++) {
//        for (int j = 0; j < result.columns(); j++) {
//            for (int k = 0; k < columns(); k++) {
//                double new_value = result.get(i, j) + get(i, k) * B.get(k, j);
//                result.set(i, j, new_value);
//            }
//        }
//    }
//    return result;
}

Matrix Matrix::hadamard_product(Matrix & B) {
    return elementwise_operation(B, E_HADAMARD);
}

Matrix Matrix::elementwise_division(Matrix & B) {
    return elementwise_operation(B, E_DIVISION);
}

Matrix Matrix::elementwise_addition(Matrix & B) {
    return elementwise_operation(B, E_ADDITION);
}

Matrix Matrix::elementwise_operation(Matrix & B, 
                                     Matrix::elementwise_operation_enum 
                                     operation_type) {
    equal_size(B);
    Matrix result = Matrix(rows(), columns());
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < columns(); j++) {
            if (operation_type == E_HADAMARD) {
                result.set(i, j, get(i, j) * B.get(i, j));
            } else if (operation_type == E_DIVISION && B.get(i, j) != 0) {
                result.set(i, j, get(i, j) / B.get(i, j));
            } else if (operation_type == E_ADDITION) {
                result.set(i, j, get(i, j) + B.get(i, j));
            }
        }
    }
    return result;
}

Matrix Matrix::elementwise_log() {
    return scalar_operation(0, S_LOG);
}

Matrix Matrix::scalar_addition(double scalar) {
    return scalar_operation(scalar, S_ADDITION);
}

Matrix Matrix::scalar_multiplication(double scalar) {
    return scalar_operation(scalar, S_MULTIPLICATION);
}

Matrix Matrix::scalar_division(double scalar) {
    return scalar_operation(scalar, S_DIVISION);
}

Matrix Matrix::scalar_minus_matrix(double scalar) {
    return scalar_operation(scalar, S_N_MINUS);
}

Matrix Matrix::scalar_operation(double scalar, 
                                Matrix::scalar_operation_enum
                                operation_type) {
    Matrix result = Matrix(rows(), columns());
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < columns(); j++) {
            if (operation_type == S_ADDITION) {
                result.set(i, j, get(i, j) + scalar);
            } else if (operation_type == S_MULTIPLICATION) {
                result.set(i, j, get(i, j) * scalar);
            } else if (operation_type == S_DIVISION && scalar != 0) {
                result.set(i, j, get(i, j) / scalar);
            } else if (operation_type == S_N_MINUS) {
                result.set(i, j, scalar - get(i, j));
            } else if (operation_type == S_LOG) {
                if (get(i, j) == 0) {
                    result.set(i, j, 0);
                } else {
                    result.set(i, j, log(get(i, j)));
                }
            }
        }
    }
    return result;
}

double Matrix::distance(Matrix & B) {
    equal_size(B);
    double total_distance = 0;
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < columns(); j++) {
            total_distance += abs(get(i, j) - B.get(i, j));
        }
    }
    return total_distance;
}

Matrix Matrix::norm() {
    Matrix result = Matrix(rows(), columns());
    double sumSquares = 0;
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < columns(); j++) {
            result.set(i, j, get(i, j));
            sumSquares += pow(get(i, j), 2);
        }
    }
    return scalar_division(sqrt(sumSquares));
}

Matrix Matrix::append_matrix(Matrix & B) {
    if (rows() != B.rows()) {
        std::cerr << "Matrices have different row counts, A: "
            << rows() << " and B: " << B.rows() << std::endl;
        // int crash = 1/0;
        exit(1);
    }
    Matrix result = Matrix(rows(), columns() + B.columns());
    for (int i = 0; i < result.rows(); i++) {
        for (int j = 0; j < result.columns(); j++) {
            if (j < columns()) {
                result.set(i, j, get(i, j));
            } else {
                result.set(i, j, B.get(i, j - columns()));
            }
        }
    }
    return result;
}

Matrix Matrix::get_column(int index) {
    if (index < 0 || index >= columns()) {
        std::cerr << "Specified index is greater than #columns - "
             << index << " > " << columns() << std::endl;
        // int crash = 1/0;
        exit(1);
    }
    Matrix result = Matrix(rows(), 1);
    for (int i = 0; i < result.rows(); i++) {
        result.set(i, 0, get(i, index));
    }
    return result;
}

Matrix Matrix::remove_column(int index) {
    if (index < 0 || index >= columns()) {
        std::cerr << "Specified index is greater than #columns - "
            << index << " > " << columns() << std::endl;
        // int crash = 1/0;
        exit(1);
    }
    Matrix result = Matrix(rows(), columns() - 1);
    for (int i = 0; i < result.rows(); i++) {
        for (int j = 0; j < result.columns(); j++) {
            if (j < index) {
                result.set(i, j, get(i, j));
            } else {
                result.set(i, j, get(i, j + 1));
            }
        }
    }
    return result;
}

// Attaches a vector of a specified value to the
// left of the matrix. 
Matrix Matrix::prepend_column(int value) {
    Matrix result = Matrix(rows(), columns() + 1);
    for (int i = 0; i < result.rows(); i++) {
        for (int j = 0; j < result.columns(); j++) {
            if (j == 0) {
                result.set(i, j, value);
            } else {
                result.set(i, j, get(i, j - 1));
            }
        }
    }
    return result;
}

// Changes an MxN matrix 'A' to 1xN vector 'B' where
// B's column values are the sum of A's values 
// in that column.
Matrix Matrix::sum_columns_to_row_vector() {
    Matrix result = Matrix(1, columns());
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < columns(); j++) {
            result.set(0, j, result.get(0, j) + get(i, j));
        }
    }
    return result;
}

// Compares 2 matrices for equality. Because of numerical
// errors, each element of the two matrices are checked to see
// if they are within 1e-12 of each other, if any entry isn't
// the matrices are said to not be equal. 
bool Matrix::equals(Matrix & B) {
    if (rows() != B.rows() || columns() != B.columns()) {
        return false;
    }
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < columns(); j++) {
            double diff = get(i, j) - B.get(i, j);
            if (diff < 0) diff *= -1;
            if (diff > 1.0e-12) return false;
        }
    }
    return true;
}

Matrix Matrix::identity(int side_length) {
    Matrix result = Matrix(side_length, side_length);
    int i = 0;
    while (i < side_length) {
        result.set(i, i, 1.0);
        i++;
    }
    return result;
}

void Matrix::display() {
    for (int i = 0; i < rows(); i++) {
        for (int j = 0; j < columns(); j++) {
            printf("%0.1f ", get(i,j));
        }
        printf("\n");
    }
    printf("------\n");
}





