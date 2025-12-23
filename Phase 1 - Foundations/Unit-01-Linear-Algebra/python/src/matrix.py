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
        
    


    
        