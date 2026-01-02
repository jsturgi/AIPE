"""
Matrix class implementation from scratch
Matrices as collections of column vectors
"""
import math
from typing import List, Union, Tuple
from src.vector import Vector

class Matrix:
    """
    A mathematical matrix with operations.
    
    Internally stored as a 2D list (list of rows).
    Can also be viewed as a collection of column vectors.

    Attributes:
        data: 2D list of values
        rows: Number of rows
        cols: Number of columns
    """
    def __init__(self, data: List[List[Union[int, float]]]):
        """
        Initialize matrix with 2D list of values.
        
        Args:
            data: List of rows, where each row is a list of numbers.
                  [[row1], [row2], ...]

        Raises:
            ValueError: If data is empty or rows have different lengths.

        Example:
            >>> m = Matrix([[1, 2], [3, 4]])
            >>> m.rows, m.cols
            (2, 2)
        """
        if not data or not data[0]:
            raise ValueError("Data Empty")
        for row in data:
            if len(row) != len(data[0]):
                raise ValueError("Jagged Array! All rows must be the same length.")

        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Return (rows, cols) tuple."""
        return (self.rows, self.cols)
    
    def __repr__(self) -> str:
        """
        String representation of matrix.

        Returns:
            Multi-line string showing matrix.
        """
        return f"Matrix({self.data})"
    
    def __str__(self) -> str:
        """

        Pretty print the matrix in readable format.

        Returns:
            Multi-line string with formatted matrix
            Example:
                [1.00, 2.00, 3.00]
                [4.00, 5.00, 6.00]
        """
        rowstrings = []
        for row in self.data:
            rstring = ", ".join(f"{num:.2f}" for num in row)
            rowstrings.append(rstring)

        return "\n".join(rowstrings)
    
    def __eq__(self, other: object) -> bool:
        """Check matrix equality."""
        if type(other) is not Matrix:
            return False
        return self.data == other.data
    
    def __getitem__(self, key: Tuple[int, int]) -> Union[int, float]:
        """
        Access element by index: m[i, j]

        Args:
            key: Tuple of (row_index, col_index)

        Returns:
            Element at that position.
        """
        return self.data[key[0]][key[1]]
    
    def __setitem__(self, key: Tuple[int, int], value: Union[int, float]):
        """Set element by index: m[i, j] = value"""
    
        self.data[key[0]][key[1]] = value
    
    def get_row(self, i: int) -> Vector:
         """
         Get row i as a Vector.
 
         Args:
             i: Row index (0-based)
 
         Returns:
             Vector containing row values.
         """
         return Vector(self.data[i])
    
    def get_column(self, j: int) -> Vector:
        """
        Get column j as a Vector.

        Args:
            j: Column index (0-based)

        Returns:
            Vector containing column values.
        """
        if j < 0:
            raise ValueError("index must be >= 0")
        if j > self.cols:
            raise ValueError("index out of bounds")
        col_values = []
        for row in self.data:
            col_values.append(row[j])
        return Vector(col_values)
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        """
        Add two matrices element-wise.

        Args:
            other: Matrix to add.

        Returns:
            New matrix that is the sum.

        Raises:
            ValueError: If shapes don't match.
        """
        if (self.cols != other.cols or self.rows != other.rows):
            raise ValueError("Inner Dimensions don't match. Invalid operation")
        return Matrix([[x+y for x,y in zip(row_a, row_b)] 
            for row_a,row_b in zip(self.data, other.data)])
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """Subtract matrices element-wise."""
        if (self.cols != other.cols or self.rows != other.rows):
            raise ValueError("Inner Dimensions don't match. Invalid operation")
        return Matrix([[x-y for x,y in zip(row_a, row_b)] 
            for row_a,row_b in zip(self.data, other.data)])
    
    def __mul__(self, scalar: Union[int, float]) -> 'Matrix':
        """
        Multiply matrix by scalar.

        Args:
            scalar: Number to multiply by.

        Returns:
            New scaled matrix.
        """
        return Matrix([[scalar * val for val in row] for row in self.data])
    
    def __rmul__(self, scalar: Union[int, float]) -> 'Matrix':
        """Allow scalar * matrix."""
        return self * scalar
    def __matmul__(self, other: Union['Matrix', Vector]) -> Union['Matrix', Vector]:
        """
        Matrix multiplication using @ operator.

        Handles both:
        - Matrix @ Matrix -> Matrix
        - Matrix @ Vector -> Vector

        Args:
            other: Matrix or Vector to multiply.

        Returns:
            Matrix (if other is Matrix) or Vector (if other is Vector).

        Raises:
            ValueError: If dimensions don't allow multiplication.
        """
        if isinstance(other, Vector):
            return self._matmul_vector(other)
        elif isinstance(other, Matrix):
            return self._matmul_matrix(other)
        else:
            raise TypeError(f"Cannot multiply Matrix with {type(other)}")   
    
    def _matmul_vector(self, v: Vector) -> Vector:
        """
        Multiply matrix by vector.

        For A (m×n) and v (n×1), result is (m×1).

        Args:
            v: Vector with dimension matching self.cols

        Returns:
            Resulting vector.

        Two equivalent views:
        1. Each result element is dot product of row with v
        2. Result is linear combination of columns weighted by v

        Hints:
            - Result has self.rows elements
            - result[i] = dot(row_i, v)
        """
        if v.dimension != self.cols:
            raise ValueError("Dimensions don't match")
        return Vector([v.dot(Vector(row)) for row in self.data])

    def _matmul_matrix(self, other: 'Matrix') -> 'Matrix':
        """
        Multiply two matrices.
    
        For A (m×n) and B (n×p), result is (m×p).
    
        Args:
            other: Matrix with rows matching self.cols
    
        Returns:
            Resulting matrix.
    
        Computation:
            result[i,j] = dot(row_i of self, col_j of other)
    
        Hints:
            - Result has self.rows rows and other.cols columns
            - Each element is a dot product
        """
        if self.cols != other.rows:
            raise ValueError("Dimensions don't match")
        columns = [self @ other.get_column(j) for j in range(other.cols)]
        return Matrix.from_column_vectors(columns)
    
    def transpose(self) -> 'Matrix':
        """
        Return transpose of matrix (rows become columns).

        Returns:
            New matrix with shape (cols, rows).

        The element at [i,j] in original becomes [j,i] in transpose.
        """
        return Matrix([self.get_column(i).components for i in range(self.cols)])
    
    @staticmethod
    def identity(n: int) -> 'Matrix':
        """
        Create n×n identity matrix.

        Args:
            n: Size of matrix.

        Returns:
            Identity matrix with 1s on diagonal, 0s elsewhere.
        """
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    
    @staticmethod
    def zeros(rows: int, cols: int) -> 'Matrix':
        """Create matrix of zeros."""
        # TODO: Return matrix filled with zeros
        return Matrix([[0]*cols for _ in range(rows)])
    
    @staticmethod
    def from_column_vectors(vectors: List[Vector]) -> 'Matrix':
        """
        Create matrix from list of column vectors.

        Args:
            vectors: List of vectors (all same dimension)

        Returns:
            Matrix where each vector is a column.

        This is the "columns view" of a matrix.
        """
        for vec in vectors:
            if vec.dimension != vectors[0].dimension:
                raise ValueError("Vectors must have same dimension")
        return Matrix([list(row) for row in zip(*[vec.components for vec in vectors])])
    
    
    

def rotation_matrix(angle_degrees: float) -> Matrix:
    """
    Create 2D rotation matrix.

    Args:
        angle_degrees: Angle to rotate counterclockwise.

    Returns:
        2x2 rotation matrix.

    Formula:
        [ cos(θ)  -sin(θ) ]
        [ sin(θ)   cos(θ) ]

    The columns are where [1,0] and [0,1] land after rotation.
    """
    radians = math.radians(angle_degrees)
    r1 = [math.cos(radians), -math.sin(radians)]
    r2 = [math.sin(radians), math.cos(radians)]
    
    return Matrix([r1,r2])

def scaling_matrix(sx: float, sy: float) -> Matrix:
    """
    Create 2D scaling matrix.

    Args:
        sx: Scale factor for x-axis.
        sy: Scale factor for y-axis.

    Returns:
        2x2 scaling matrix.

    Formula:
        [ sx  0  ]
        [ 0   sy ]
    """
    return Matrix([[sx, 0], [0,sy]])

def shear_matrix(kx: float = 0, ky: float = 0) -> Matrix:
    """
    Create 2D shearing matrix.

    Args:
        kx: Horizontal shear factor.
        ky: Vertical shear factor.

    Returns:
        2x2 shearing matrix.

    Formula:
        [ 1   kx ]
        [ ky  1  ]
    """
    return Matrix([[1, kx], [ky, 1]])
 
def reflection_matrix(axis: str) -> Matrix:
     """
     Create 2D reflection matrix.
 
     Args:
         axis: 'x' for reflection over x-axis,
               'y' for reflection over y-axis,
               'origin' for reflection through origin.
 
     Returns:
         2x2 reflection matrix.
     """
     match axis:
         case "x":
             return Matrix([[1,0], [0,-1]])
         case "y":
             return Matrix([[-1,0], [0,1]])
         case "origin":
             return Matrix([[-1,0], [0,-1]])
         case _:
             raise ValueError("Axis mismatch: only x, y, and origin supported.")

def projection_matrix(onto: str) -> Matrix:
    """
    Create 2D projection matrix.

    Args:
        onto: 'x' for projection onto x-axis,
              'y' for projection onto y-axis.

    Returns:
        2x2 projection matrix.

    Note: Projection loses information (determinant = 0).
    """
    match onto:
        case "x":
            return Matrix([[1,0], [0,0]])
        case "y":
            return Matrix([[0,0], [0,1]])
        case _:
            raise ValueError("Only x and y supported")

def compose(*matrices: Matrix) -> Matrix:
    """
    Compose multiple transformations (apply left to right).

    Args:
        matrices: Matrices to compose, applied in order given.

    Returns:
        Single matrix representing composed transformation.

    Example:
        compose(A, B, C) applies A first, then B, then C.
        Mathematically: C @ B @ A
    """
    toReturn = matrices[-1]
    for m in reversed(matrices[:-1]):
        toReturn = toReturn @ m
    return toReturn

def determinant_2x2(matrix: Matrix) -> float:
    """
    Compute determinant of a 2x2 matrix.

    Args:
        matrix: A 2x2 Matrix object.

    Returns:
        The determinant (ad - bc).

    Raises:
        ValueError: If matrix is not 2x2.

    Formula:
        [ a  b ]
        [ c  d ] → ad - bc
    """
    if matrix.rows != matrix.cols:
        raise ValueError("Not a square matrix. Determinant cannot be computed.")
    if matrix.rows != 2:
        raise ValueError("Only supports 2x2 Matrices.")
    a,b = matrix.data[0]
    c,d = matrix.data[1]

    return a*d - b*c



def determinant_3x3(matrix: Matrix) -> float:
    """
    Compute determinant of a 3x3 matrix using cofactor expansion.

    Args:
        matrix: A 3x3 Matrix object.

    Returns:
        The determinant.

    Raises:
        ValueError: If matrix is not 3x3.

    Formula (expansion along first row):
        a(ei - fh) - b(di - fg) + c(dh - eg)

    where:
        [ a  b  c ]
        [ d  e  f ]
        [ g  h  i ]
    """
    a,b,c = matrix.data[0]
    d,e,f = matrix.data[1]
    g,h,i = matrix.data[2]
    return a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)

def minor(matrix: Matrix, row: int, col: int) -> Matrix:
    """
    Get the minor matrix by removing specified row and column.

    Args:
        matrix: The original matrix.
        row: Row index to remove (0-indexed).
        col: Column index to remove (0-indexed).

    Returns:
        New matrix with one less row and column.

    Example:
        minor([[1,2,3],[4,5,6],[7,8,9]], 0, 0) → [[5,6],[8,9]]
    """
    toReturn = []
    for i in range(matrix.rows):
        toAppend = []
        if i != row:
            toAppend = matrix.get_row(i).components
            toAppend.pop(col)
            toReturn.append(toAppend)
    return Matrix(toReturn)

def cofactor(matrix: Matrix, row: int, col: int) -> float:
    """
    Compute the cofactor at position (row, col).

    Cofactor = (-1)^(row+col) * det(minor)

    Args:
        matrix: The original matrix.
        row: Row index.
        col: Column index.

    Returns:
        The cofactor value.
    """
    matMinor = minor(matrix, row, col)
    minorDet = determinant(matMinor)
    return (-1)**(row+col) * minorDet
    
def determinant(matrix: Matrix):
        """
        Computes determinant
        """
       
        if matrix.cols != matrix.rows:
            raise ValueError("Not a square matrix. Determinant cannot be computed.")
        match matrix.rows:
            case 1:
                return matrix.data[0][0]
            case 2:
                a,b = matrix.data[0]
                c,d = matrix.data[1]
                return a*d - b*c
            case 3:
                a,b,c = matrix.data[0]
                d,e,f = matrix.data[1]
                g,h,i = matrix.data[2]
                return a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
            case _: # row reduction
                swaps = 0
                matrix_cp = [row.copy() for row in matrix.data]
                for i in range(matrix.cols-1):
                    if not matrix_cp[i][i]: #zero pivot
                        row_swap = None
                        for k in range(i+1, matrix.rows): #search rows below pivot
                            if matrix_cp[k][i]:
                                row_swap = k
                                break
                        if not row_swap:
                            return 0 # all zeros, det 0
                        matrix_cp[row_swap], matrix_cp[i] = matrix_cp[i], matrix_cp[row_swap]
                        swaps+=1
                    for j in range(i+1, matrix.rows):
                        multiplier = matrix_cp[j][i] / matrix_cp[i][i]
                        for col in range(matrix.cols):
                            matrix_cp[j][col] = matrix_cp[j][col] - multiplier*matrix_cp[i][col]
                det = 1
                for i in range(matrix.cols):
                    det = det * matrix_cp[i][i]
                if swaps % 2 == 0:
                    return det
                else:
                    return det*-1
def is_invertible(matrix: Matrix) -> bool:
    """
    Check if a matrix is invertible.

    A matrix is invertible if and only if its determinant is non-zero.

    Args:
        matrix: A square matrix.

    Returns:
        True if invertible, False otherwise.
    """
    det = determinant(matrix)
    tolerance = 1e-10
    return abs(det) > tolerance

def cofactor_matrix(matrix: Matrix) -> Matrix:
    """
    Compute the cofactor matrix.

    The cofactor matrix C has C[i,j] = cofactor(matrix, i, j).

    Args:
        matrix: A square matrix.

    Returns:
        The cofactor matrix.
    """
    mat_copy = [row.copy() for row in matrix.data]
    mat_copy = [[cofactor(matrix, i, j) for j in range(matrix.cols)] for i in range(matrix.rows)]
    return Matrix(mat_copy)

def adjugate(matrix: Matrix) -> Matrix:
    """
    Compute the adjugate (classical adjoint) matrix.

    adjugate(A) = transpose(cofactor_matrix(A))

    Args:
        matrix: A square matrix.

    Returns:
        The adjugate matrix.

    Note: A⁻¹ = adjugate(A) / det(A)
    """
    return cofactor_matrix(matrix).transpose()

def inverse_adjugate(matrix: Matrix) -> Matrix:
    """
    Compute matrix inverse using the adjugate method.

    A⁻¹ = (1/det(A)) * adjugate(A)

    Args:
        matrix: An invertible square matrix.

    Returns:
        The inverse matrix.

    Raises:
        ValueError: If matrix is not invertible (det = 0).

    Complexity: O(n! * n²) - not efficient for large matrices.
    """
    det = determinant(matrix)
    tolerance = 1e-10
    
    if abs(det) < tolerance:
        raise ValueError("Matrix is not invertible (det = 0")
    
    return ((1/det) * adjugate(matrix))

def inverse_2x2(matrix: Matrix) -> Matrix:
    """
    Compute 2x2 matrix inverse using direct formula.

    For A = [[a,b],[c,d]], A⁻¹ = (1/det) * [[d,-b],[-c,a]]

    Args:
        matrix: A 2x2 invertible matrix.

    Returns:
        The inverse matrix.

    Raises:
        ValueError: If not 2x2 or not invertible.
    """
    if (matrix.rows != 2) and (matrix.cols != 2):
        raise ValueError("Matrix must be 2x2.")
    if not is_invertible(matrix):
        raise ValueError("Matrix is not invertible")
    det = determinant_2x2(matrix)
    a,b = matrix.data[0]
    c,d = matrix.data[1]
    inverse = [[d,-b],[-c,a]]
    return (1/det) * Matrix(inverse)
    
def swap_rows(matrix: Matrix, i: int, j: int) -> Matrix:
    """
    Swap two rows in a matrix (returns new matrix).

    Args:
        matrix: The matrix.
        i, j: Row indices to swap.

    Returns:
        New matrix with rows swapped.
    """
    new_data = [row[:] for row in matrix.data]

    # Swap rows i and j
    new_data[i], new_data[j] = new_data[j], new_data[i]

    return Matrix(new_data)


def scale_row(matrix: Matrix, i: int, scalar: float) -> Matrix:
    """
    Multiply a row by a scalar (returns new matrix).

    Args:
        matrix: The matrix.
        i: Row index.
        scalar: Value to multiply by.

    Returns:
        New matrix with row scaled.
    """
    new_data = [row[:] for row in matrix.data]
    
    new_data[i] = [val * scalar for val in new_data[i]]
    
    return Matrix(new_data)


def add_row_multiple(matrix: Matrix, target: int, source: int, scalar: float) -> Matrix:
    """
    Add a multiple of one row to another (returns new matrix).

    row[target] = row[target] + scalar * row[source]

    Args:
        matrix: The matrix.
        target: Row to modify.
        source: Row to add from.
        scalar: Multiple to add.

    Returns:
        New matrix with row operation applied.
    """
    new_data = [row[:] for row in matrix.data]
    
    new_data[target] = [
        new_data[target][col] + scalar * new_data[source][col]
        for col in range(len(new_data[target]))
    ]
    return Matrix(new_data)

def augment(left: Matrix, right: Matrix) -> Matrix:
    """
    Create augmented matrix [left | right].

    Args:
        left: Left matrix.
        right: Right matrix (must have same number of rows).

    Returns:
        Augmented matrix with columns concatenated.
    """
    if left.rows != right.rows:
        raise ValueError("Rows don't match")
    return Matrix([left.get_row(i).components + right.get_row(i).components for i in range(left.rows) ])
    
def inverse_gauss_jordan(matrix: Matrix) -> Matrix:
    """
    Compute matrix inverse using Gauss-Jordan elimination.

    Algorithm:
    1. Augment [A | I]
    2. Apply row operations to transform A to I
    3. The right side becomes A⁻¹

    Args:
        matrix: An invertible square matrix.

    Returns:
        The inverse matrix.

    Raises:
        ValueError: If matrix is not invertible.

    Complexity: O(n³) - much better than adjugate method!
    """
    n = matrix.rows
    identity = matrix.identity(n)
    aug = augment(matrix, identity)
    
    for j in range(n):
        pivot_row = None
        for row in range(j,n):
            if aug.data[row][j] != 0:
                pivot_row = row
                break
        if pivot_row is None:
            raise ValueError("Matrix is singular")
        if pivot_row != j:
            aug = swap_rows(aug, j, pivot_row)
        aug = scale_row(aug,j,1/aug.data[j][j])
        for row in range(j+1,n):
            aug = add_row_multiple(aug, row, j, -aug.data[row][j])
    for j in range (n-1, 0, -1):
        for i in range(0, j):
            aug = add_row_multiple(aug, i, j, -aug.data[i][j])
    return Matrix([row[n:] for row in aug.data])
 
def inverse(matrix: Matrix) -> Matrix:
    """
    Compute matrix inverse (dispatcher function).

    Uses efficient method based on matrix size:
    - 2x2: Direct formula
    - Larger: Gauss-Jordan elimination

    Args:
        matrix: An invertible square matrix.

    Returns:
        The inverse matrix.

    Raises:
        ValueError: If matrix is not square or not invertible.
    """
    if (matrix.rows != matrix.cols):
        raise ValueError("Matrix not square")
    if matrix.rows == 2:
        return inverse_2x2(matrix)
    else:
        return inverse_gauss_jordan(matrix)

