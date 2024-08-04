import numpy as np

def matrix_multiply(matrix1, matrix2):
    """
    Multiply two matrices.

    :param matrix1: First matrix (2D NumPy array)
    :param matrix2: Second matrix (2D NumPy array)
    :return: Result of matrix multiplication (2D NumPy array)
    """
    return np.matmul(matrix1, matrix2)

def matrix_transpose(matrix):
    """
    Compute the transpose of a matrix.

    :param matrix: Input matrix (2D NumPy array)
    :return: Transposed matrix (2D NumPy array)
    """
    return np.transpose(matrix)

def matrix_determinant(matrix):
    """
    Calculate the determinant of a square matrix.

    :param matrix: Square matrix (2D NumPy array)
    :return: Determinant of the matrix (float)
    """
    return np.linalg.det(matrix)

def matrix_inverse(matrix):
    """
    Compute the inverse of a square matrix.

    :param matrix: Square matrix (2D NumPy array)
    :return: Inverse of the matrix (2D NumPy array)
    """
    return np.linalg.inv(matrix)

def solve_linear_system(coefficients, constants):
    """
    Solve a system of linear equations.

    :param coefficients: Coefficient matrix (2D NumPy array)
    :param constants: Constants vector (1D NumPy array)
    :return: Solution vector (1D NumPy array)
    """
    return np.linalg.solve(coefficients, constants)

def eigenvalues(matrix):
    """
    Compute the eigenvalues of a square matrix.

    :param matrix: Square matrix (2D NumPy array)
    :return: Array of eigenvalues (1D NumPy array)
    """
    return np.linalg.eigvals(matrix)

def eigenvectors(matrix):
    """
    Compute the eigenvectors of a square matrix.

    :param matrix: Square matrix (2D NumPy array)
    :return: Tuple (eigenvalues, eigenvectors) where eigenvectors is a 2D NumPy array
    """
    return np.linalg.eig(matrix)

def vector_norm(vector):
    """
    Compute the Euclidean norm (L2 norm) of a vector.

    :param vector: Input vector (1D NumPy array)
    :return: Euclidean norm of the vector (float)
    """
    return np.linalg.norm(vector)

def matrix_rank(matrix):
    """
    Compute the rank of a matrix.

    :param matrix: Input matrix (2D NumPy array)
    :return: Rank of the matrix (int)
    """
    return np.linalg.matrix_rank(matrix)

def orthogonalize(matrix):
    """
    Compute an orthogonal basis for the range of a matrix using the Gram-Schmidt process.

    :param matrix: Input matrix (2D NumPy array)
    :return: Orthogonalized matrix (2D NumPy array)
    """
    q, r = np.linalg.qr(matrix)
    return q
