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
        # Guard against empty input
        if not data or not data[0]:
            raise ValueError("Data Empty")

        # Ensure all rows have same length (no jagged arrays)
        for row in data:
            if len(row) != len(data[0]):
                raise ValueError("Jagged Array! All rows must be the same length.")

        self.data = data
        self.rows = len(data)      # m dimension
        self.cols = len(data[0])   # n dimension
    
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
        # Bounds checking
        if j < 0:
            raise ValueError("index must be >= 0")
        if j > self.cols:
            raise ValueError("index out of bounds")

        # Extract j-th element from each row to form column vector
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
        # Matrix addition requires identical dimensions
        if (self.cols != other.cols or self.rows != other.rows):
            raise ValueError("Inner Dimensions don't match. Invalid operation")

        # Element-wise addition: zip rows together, then zip elements within rows
        return Matrix([[x+y for x,y in zip(row_a, row_b)]
            for row_a,row_b in zip(self.data, other.data)])
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """Subtract matrices element-wise."""
        if (self.cols != other.cols or self.rows != other.rows):
            raise ValueError("Inner Dimensions don't match. Invalid operation")

        # Element-wise subtraction
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
        # Dispatch to appropriate multiplication method based on operand type
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
        # Vector dimension must match number of columns
        if v.dimension != self.cols:
            raise ValueError("Dimensions don't match")

        # Each output element is dot product of corresponding row with input vector
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
        # Inner dimensions must match: (m×n) @ (n×p) = (m×p)
        if self.cols != other.rows:
            raise ValueError("Dimensions don't match")

        # Column-wise approach: each result column is A times corresponding column of B
        # This leverages matrix-vector multiplication we already implemented
        columns = [self @ other.get_column(j) for j in range(other.cols)]
        return Matrix.from_column_vectors(columns)
    
    def transpose(self) -> 'Matrix':
        """
        Return transpose of matrix (rows become columns).

        Returns:
            New matrix with shape (cols, rows).

        The element at [i,j] in original becomes [j,i] in transpose.
        """
        # Each column of original becomes a row in transpose
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
        # Diagonal elements (i==j) are 1, all others are 0
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])
    
    @staticmethod
    def zeros(rows: int, cols: int) -> 'Matrix':
        """Create matrix of zeros."""
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
        # All vectors must have same dimension to form valid matrix
        for vec in vectors:
            if vec.dimension != vectors[0].dimension:
                raise ValueError("Vectors must have same dimension")

        # zip(*...) transposes: columns become rows
        # Each vector's components become a column in the resulting matrix
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
    # Convert degrees to radians for trig functions
    radians = math.radians(angle_degrees)

    # Row 1: how x-component transforms
    r1 = [math.cos(radians), -math.sin(radians)]
    # Row 2: how y-component transforms
    r2 = [math.sin(radians), math.cos(radians)]

    return Matrix([r1, r2])

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
    # Start with last matrix (rightmost in multiplication order)
    toReturn = matrices[-1]

    # Multiply from right to left: result = Mn @ ... @ M2 @ M1
    # This gives left-to-right application order when transforming vectors
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
    # Determinant only defined for square matrices
    if matrix.rows != matrix.cols:
        raise ValueError("Not a square matrix. Determinant cannot be computed.")
    if matrix.rows != 2:
        raise ValueError("Only supports 2x2 Matrices.")

    # Extract elements
    a, b = matrix.data[0]
    c, d = matrix.data[1]

    # Cross-multiply: main diagonal minus anti-diagonal
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
    # Extract all 9 elements
    a, b, c = matrix.data[0]
    d, e, f = matrix.data[1]
    g, h, i = matrix.data[2]

    # Cofactor expansion along first row:
    # +a * det(2x2 minor) - b * det(2x2 minor) + c * det(2x2 minor)
    return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

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
        # Skip the row we're removing
        if i != row:
            # Get row as list and remove the specified column
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
    # Get the (n-1)×(n-1) submatrix with row and col removed
    matMinor = minor(matrix, row, col)

    # Compute determinant of the minor
    minorDet = determinant(matMinor)

    # Apply checkerboard sign pattern: + - + - ...
    # Sign is positive when (row+col) is even, negative when odd
    return (-1)**(row+col) * minorDet
    
def determinant(matrix: Matrix):
        """
        Computes determinant using optimized method based on matrix size.

        Uses direct formulas for 1x1, 2x2, 3x3 matrices.
        Uses row reduction (Gaussian elimination) for larger matrices - O(n³).
        """
        # Determinant only defined for square matrices
        if matrix.cols != matrix.rows:
            raise ValueError("Not a square matrix. Determinant cannot be computed.")

        match matrix.rows:
            case 1:
                # 1x1: determinant is the single element
                return matrix.data[0][0]

            case 2:
                # 2x2: ad - bc formula
                a, b = matrix.data[0]
                c, d = matrix.data[1]
                return a*d - b*c

            case 3:
                # 3x3: cofactor expansion along first row
                a, b, c = matrix.data[0]
                d, e, f = matrix.data[1]
                g, h, i = matrix.data[2]
                return a*(e*i - f*h) - b*(d*i - f*g) + c*(d*h - e*g)

            case _:
                # n×n (n > 3): Use row reduction to upper triangular form
                # det(A) = (-1)^swaps * product of diagonal elements
                swaps = 0
                matrix_cp = [row.copy() for row in matrix.data]

                # Forward elimination: reduce to upper triangular
                for i in range(matrix.cols - 1):
                    # Handle zero pivot by swapping with row below
                    if not matrix_cp[i][i]:
                        row_swap = None
                        # Search for non-zero element in column below pivot
                        for k in range(i + 1, matrix.rows):
                            if matrix_cp[k][i]:
                                row_swap = k
                                break

                        if not row_swap:
                            # Entire column is zero -> det = 0
                            return 0

                        # Swap rows (changes sign of determinant)
                        matrix_cp[row_swap], matrix_cp[i] = matrix_cp[i], matrix_cp[row_swap]
                        swaps += 1

                    # Eliminate entries below pivot
                    for j in range(i + 1, matrix.rows):
                        multiplier = matrix_cp[j][i] / matrix_cp[i][i]
                        for col in range(matrix.cols):
                            matrix_cp[j][col] = matrix_cp[j][col] - multiplier * matrix_cp[i][col]

                # Determinant = product of diagonal (for upper triangular matrix)
                det = 1
                for i in range(matrix.cols):
                    det = det * matrix_cp[i][i]

                # Each row swap negates the determinant
                if swaps % 2 == 0:
                    return det
                else:
                    return det * -1
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

    # Matrix is invertible iff determinant is non-zero
    # Use tolerance to handle floating point precision
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
    # Build matrix where each element is replaced by its cofactor
    # C[i,j] = (-1)^(i+j) * det(minor(i,j))
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
    # Adjugate is the transpose of the cofactor matrix
    # Used in the formula: A⁻¹ = adj(A) / det(A)
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

    # Check invertibility
    if abs(det) < tolerance:
        raise ValueError("Matrix is not invertible (det = 0")

    # A⁻¹ = (1/det(A)) * adj(A)
    # Each element of adjugate is scaled by 1/det
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
    # Validate 2x2 size
    if (matrix.rows != 2) and (matrix.cols != 2):
        raise ValueError("Matrix must be 2x2.")
    if not is_invertible(matrix):
        raise ValueError("Matrix is not invertible")

    det = determinant_2x2(matrix)

    # Extract elements
    a, b = matrix.data[0]
    c, d = matrix.data[1]

    # Direct formula: swap a,d and negate b,c, then scale by 1/det
    # [[a,b],[c,d]]⁻¹ = (1/det) * [[d,-b],[-c,a]]
    inverse = [[d, -b], [-c, a]]
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
    # Deep copy to avoid mutating original
    new_data = [row[:] for row in matrix.data]

    # Swap rows i and j using tuple unpacking
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
    # Deep copy to avoid mutating original
    new_data = [row[:] for row in matrix.data]

    # Multiply every element in row i by scalar
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
    # Deep copy to avoid mutating original
    new_data = [row[:] for row in matrix.data]

    # target_row = target_row + scalar * source_row
    # This is the key operation for elimination (use scalar = -multiplier)
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
    # Must have same number of rows to augment
    if left.rows != right.rows:
        raise ValueError("Rows don't match")

    # Concatenate each row: [left_row | right_row]
    return Matrix([left.get_row(i).components + right.get_row(i).components for i in range(left.rows)])
    
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

    # Create augmented matrix [A | I]
    # Goal: transform to [I | A⁻¹]
    identity = matrix.identity(n)
    aug = augment(matrix, identity)

    # === FORWARD ELIMINATION ===
    # Transform left side to upper triangular with 1s on diagonal
    for j in range(n):
        # Find pivot: first non-zero element in column j, rows >= j
        pivot_row = None
        for row in range(j, n):
            if aug.data[row][j] != 0:
                pivot_row = row
                break

        # No pivot found means matrix is singular (not invertible)
        if pivot_row is None:
            raise ValueError("Matrix is singular")

        # Swap pivot row to diagonal position if needed
        if pivot_row != j:
            aug = swap_rows(aug, j, pivot_row)

        # Scale row to make pivot = 1
        aug = scale_row(aug, j, 1/aug.data[j][j])

        # Eliminate entries below pivot (make them 0)
        for row in range(j + 1, n):
            aug = add_row_multiple(aug, row, j, -aug.data[row][j])

    # === BACKWARD ELIMINATION ===
    # Eliminate entries above each pivot (already 1s on diagonal)
    # Work from bottom-right to top-left
    for j in range(n - 1, 0, -1):
        for i in range(0, j):
            # Eliminate entry at (i, j) by subtracting multiple of row j
            aug = add_row_multiple(aug, i, j, -aug.data[i][j])

    # Extract right half of augmented matrix (the inverse)
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
    # Only square matrices can be inverted
    if (matrix.rows != matrix.cols):
        raise ValueError("Matrix not square")

    # Use direct formula for 2x2 (slightly faster)
    # Use Gauss-Jordan for all other sizes
    if matrix.rows == 2:
        return inverse_2x2(matrix)
    else:
        return inverse_gauss_jordan(matrix)

def copy_matrix(matrix: Matrix) -> Matrix:
    """Create a deep copy of a matrix."""
    # row[:] creates a shallow copy of each row list
    return Matrix([row[:] for row in matrix.data])

def forward_elimination(augmented: Matrix) -> tuple[Matrix, int]:
    """
    Perform forward elimination to get row echelon form.

    Args:
        augmented: The augmented matrix [A | b].

    Returns:
        Tuple of (row echelon form, rank).

    Algorithm:
        For each column j (as pivot column):
            1. Find pivot: largest |value| in column j, rows >= j
            2. Swap pivot row to row j
            3. If pivot is 0 (or near 0), skip column (rank doesn't increase)
            4. Eliminate below: make all entries below pivot zero
    """
    A = copy_matrix(augmented)
    rows = A.rows
    # Only process coefficient columns (exclude augmented b column)
    pivot_cols = min(A.rows, A.cols - 1)
    rank = 0

    for j in range(pivot_cols):
        # === PARTIAL PIVOTING ===
        # Find largest absolute value in column j (rows >= j)
        # This improves numerical stability
        pivot = 0
        pivot_row = None
        for row in range(j, rows):
            if abs(A.data[row][j]) > abs(pivot):
                pivot = A.data[row][j]
                pivot_row = row

        # Non-zero pivot found means we have a pivot column
        if pivot:
            rank += 1

        # If we found a valid pivot, perform elimination
        if pivot_row is not None:
            # Swap largest element to pivot position
            A = swap_rows(A, j, pivot_row)

            # Eliminate entries below pivot
            for row in range(j + 1, rows):
                # multiplier = entry / pivot, subtract to zero out entry
                A = add_row_multiple(A, row, j, -A.data[row][j] / A.data[j][j])

    return (A, rank)

def back_substitution(row_echelon: Matrix, rank: int) -> list[float] | None:
    """
    Perform back substitution on row echelon form.

    Args:
        row_echelon: Matrix in row echelon form [A | b].
        rank: The rank of the coefficient matrix.

    Returns:
        Solution vector if unique solution exists.
        None if no solution or infinite solutions.

    Algorithm:
        Starting from last pivot row, work upward:
        1. x[i] = (b[i] - sum of known terms) / pivot
    """
    rows = row_echelon.rows
    cols = row_echelon.cols
    A = copy_matrix(row_echelon)
    n = cols - 1  # Number of variables (last column is b)
    x = [0.0] * n  # Solution vector

    # Check for inconsistency: [0 0 ... 0 | nonzero] means no solution
    for row in row_echelon.data:
        if all(abs(val) < 1e-10 for val in row[:-1]) and abs(row[-1]) > 1e-10:
            return None  # Inconsistent system

    # Underdetermined system: more unknowns than equations with non-zero rows
    if rank < n:
        return None  # Infinite solutions

    # Back substitution: solve from bottom row up
    # For row i: a[i,i]*x[i] + a[i,i+1]*x[i+1] + ... = b[i]
    # So: x[i] = (b[i] - sum(a[i,j]*x[j] for j>i)) / a[i,i]
    for i in range(rank - 1, -1, -1):
        # b[i] is in last column, subtract known terms, divide by pivot
        x[i] = (row_echelon.data[i][-1] - sum(A.data[i][j] * x[j]
            for j in range(i + 1, n))) / A.data[i][i]

    return x

def solve_system(A: Matrix, b: list[float]) -> dict:
    """
    Solve the linear system Ax = b.

    Args:
        A: Coefficient matrix (m x n).
        b: Right-hand side vector (m elements).

    Returns:
        Dictionary with:
            'status': 'unique', 'infinite', or 'inconsistent'
            'solution': The solution vector (if unique)
            'rank': Rank of A
            'augmented_rref': The reduced row echelon form

    Example:
        A = [[2, 3], [1, -2]]
        b = [8, -3]
        result = solve_system(A, b)
        # result['solution'] = [1, 2]  (x=1, y=2)
    """
    # Validate dimensions
    if A.rows != len(b):
        raise ValueError(f"Dimension mismatch: A has {A.rows} rows but b has {len(b)} elements")

    # Create augmented matrix [A | b]
    b_matrix = Matrix([[val] for val in b])
    augmented = augment(A, b_matrix)

    # Perform forward elimination
    row_echelon, rank = forward_elimination(augmented)

    n = A.cols  # Number of variables

    # Check for inconsistency: row with all zeros in A part but non-zero in b part
    for row in row_echelon.data:
        if all(abs(val) < 1e-10 for val in row[:-1]) and abs(row[-1]) > 1e-10:
            return {
                'status': 'inconsistent',
                'solution': None,
                'rank': rank,
                'augmented_rref': row_echelon
            }

    # Check for infinite solutions: rank < number of variables
    if rank < n:
        return {
            'status': 'infinite',
            'solution': None,
            'rank': rank,
            'augmented_rref': row_echelon
        }

    # Unique solution: perform back substitution
    solution = back_substitution(row_echelon, rank)

    return {
        'status': 'unique',
        'solution': solution,
        'rank': rank,
        'augmented_rref': row_echelon
    }
    
def gaussian_elimination(A: Matrix, b: list[float]) -> list[float]:
    """
    Solve Ax = b using Gaussian elimination with partial pivoting.

    Args:
        A: Square coefficient matrix.
        b: Right-hand side vector.

    Returns:
        Solution vector x.

    Raises:
        ValueError: If system has no unique solution.

    This is a convenience function that calls solve_system
    and raises if solution is not unique.
    """
    # Delegate to solve_system for the heavy lifting
    result = solve_system(A, b)

    # Only return solution for unique case; raise for infinite/inconsistent
    if result['status'] != 'unique':
        raise ValueError(f"System has {result['status']} solution(s)")
    return result['solution']

def rank(matrix: Matrix) -> int:
    """
    Compute the rank of a matrix.

    Rank = number of linearly independent rows
         = number of pivots in row echelon form

    Args:
        matrix: Any matrix.

    Returns:
        The rank (integer).
    """
    # Add zero column to make it like an augmented matrix
    augmented = augment(matrix, Matrix([[0] for _ in range(matrix.rows)]))
    _, r = forward_elimination(augmented)
    return r
    
def is_consistent(A: Matrix, b: list[float]) -> bool:
    """
    Check if the system Ax = b has at least one solution.

    A system is consistent if rank(A) = rank([A|b]).

    Args:
        A: Coefficient matrix.
        b: Right-hand side vector.

    Returns:
        True if at least one solution exists.
    """
    # Compute rank a
    rankA = rank(A) 
    # compute rank Ab
    rankAb = solve_system(A,b)['rank']
    return rankA == rankAb

def solution_type(A: Matrix, b: list[float]) -> str:
    """
    Determine the type of solution for Ax = b.

    Returns:
        'unique': Exactly one solution
        'infinite': Infinitely many solutions
        'inconsistent': No solution

    Logic:
        - If rank(A) < rank([A|b]): inconsistent
        - If rank(A) = rank([A|b]) = n (variables): unique
        - If rank(A) = rank([A|b]) < n: infinite
    """
    rankA = rank(A)
    rankAb = solve_system(A,b)['rank']
    if rankA < rankAb:
        return 'inconsistent'
    if rankA == rankAb == A.cols:
        return 'unique'
    if rankA == rankAb and rankA < A.cols:
        return 'infinite'

def solve_with_inverse(A: Matrix, b: list[float]) -> list[float]:
    """
    Solve Ax = b using matrix inverse: x = A⁻¹b.

    Args:
        A: Invertible square matrix.
        b: Right-hand side vector.

    Returns:
        Solution vector x.

    Note: This is less efficient than Gaussian elimination
    but useful for solving multiple systems with same A.
    """
    # Compute A inverse (raises if singular)
    A_inv = inverse(A)

    # x = A⁻¹ @ b: matrix-vector multiplication gives solution
    result = A_inv @ Vector(b)
    return result.components

def lu_decomposition(matrix: Matrix) -> tuple[Matrix, Matrix]:
    """
    Compute LU decomposition without pivoting.

    A = L @ U where:
    - L is lower triangular with 1s on diagonal
    - U is upper triangular

    Args:
        matrix: Square matrix to decompose.

    Returns:
        Tuple (L, U).

    Raises:
        ValueError: If matrix is not square.
        ValueError: If decomposition fails (needs pivoting).

    Algorithm (Doolittle):
        For k = 0 to n-1:
            For j = k to n-1:
                U[k,j] = A[k,j] - sum(L[k,s] * U[s,j] for s < k)
            For i = k+1 to n-1:
                L[i,k] = (A[i,k] - sum(L[i,s] * U[s,k] for s < k)) / U[k,k]

    Note: This version doesn't pivot. For numerical stability,
    use plu_decomposition instead.
    """
    n = matrix.rows
    A = [row[:] for row in matrix.data]
    if matrix.rows != matrix.cols:
        raise ValueError("Matrix must be square")

    # Initialize L as identity, U as zeros
    L = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    U = [[0.0 for j in range(n)] for i in range(n)]
    
    for k in range(n):
        for j in range(k,n):
            U[k][j] = A[k][j] #k of U is k of A after elimination
        if abs(A[k][k]) < 1e-10: #zero pivot
            raise ValueError("Zero Pivot encountered. Needs pivoting")
        for i in range(k+1, n): # i > k
            L[i][k] = A[i][k] / A[k][k] #L[row][col] where row > col
            for j in range(k,n):
                A[i][j] = A[i][j] - L[i][k] * A[k][j]
    return (Matrix(L), Matrix(U))
    
def plu_decomposition(matrix: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    """
    Compute PLU decomposition with partial pivoting.

    PA = LU where:
    - P is permutation matrix (row swaps)
    - L is lower triangular with 1s on diagonal
    - U is upper triangular

    Args:
        matrix: Square matrix to decompose.

    Returns:
        Tuple (P, L, U).

    Raises:
        ValueError: If matrix is singular.

    Algorithm:
        Modified Doolittle with row pivoting at each step.
        P starts as identity and records all swaps.
    """
    n = matrix.rows
    if matrix.rows != matrix.cols:
        raise ValueError("Matrix must be square")

    # Working copy of A
    A = [row[:] for row in matrix.data]

    # Initialize P as identity (stored as permutation vector)
    perm = list(range(n))

    # Initialize L and U
    L = [[0.0 for j in range(n)] for i in range(n)]
    U = [[0.0 for j in range(n)] for i in range(n)]
    
    for k in range(n):
        pivot = 0
        pivot_row = None
        for row in range(k, n):
            if abs(A[row][k]) > abs(pivot):
                pivot = A[row][k]
                pivot_row = row
        if abs(pivot) < 1e-10:
            raise ValueError("Matrix is Singular")
        if (pivot_row != k) and (pivot_row is not None):
            A[k], A[pivot_row] = A[pivot_row], A[k]
            perm[k], perm[pivot_row] = perm[pivot_row], perm[k]
            L[k], L[pivot_row] = L[pivot_row], L[k]
        L[k][k]= 1.0
            
        
        for j in range(k,n):
            U[k][j] = A[k][j] #k of U is k of A after elimination
       
        for i in range(k+1, n): # i > k
            L[i][k] = A[i][k] / A[k][k] #L[row][col] where row > col
            for j in range(k,n):
                A[i][j] = A[i][j] - L[i][k] * A[k][j]
    perm = permutation_matrix(perm)
    return (perm, Matrix(L), Matrix(U))


    # TODO: For each column k:
    #   1. Find pivot (max |A[i,k]| for i >= k)
    #   2. Swap rows in A and perm
    #   3. Check for zero pivot (singular)
    #   4. Compute L entries for column k
    #   5. Eliminate below pivot
    #   6. U[k,:] = current row k of A

    # TODO: Build permutation matrix from perm vector
    # TODO: Extract L and U

    
        
            
            
        

    # TODO: Implement Doolittle algorithm
    # For each column k:
    #   Compute U[k,j] for j >= k
    #   Compute L[i,k] for i > k
    #   Check for zero pivot (would need row swap)

    pass

def permutation_matrix(perm: list[int]) -> Matrix:
    """
    Create permutation matrix from permutation vector.

    Args:
        perm: List where perm[i] = j means row i of P has 1 in column j.

    Returns:
        Permutation matrix P.

    Example:
        perm = [1, 0, 2] creates:
        [[0, 1, 0],
         [1, 0, 0],
         [0, 0, 1]]
    """
    n = len(perm)
    P = [[1 if perm[i] == j else 0 for j in range(n)] for i in range(n)]
    return Matrix(P)

def forward_substitution(L: Matrix, b: list[float]) -> list[float]:
    """
    Solve Ly = b where L is lower triangular.

    Args:
        L: Lower triangular matrix (with 1s on diagonal).
        b: Right-hand side vector.

    Returns:
        Solution vector y.

    Algorithm:
        y[0] = b[0] / L[0,0]  (but L[0,0] = 1)
        y[i] = (b[i] - sum(L[i,j] * y[j] for j < i)) / L[i,i]
    """
    n = len(b)
    y = [0.0] * n
    
    for i in range(n):
        y[i] = b[i] - sum(L.data[i][j] *y[j] for j in range(i))
    return y

def backward_substitution(U: Matrix, y: list[float]) -> list[float]:
    """
    Solve Ux = y where U is upper triangular.

    Args:
        U: Upper triangular matrix.
        y: Right-hand side vector.

    Returns:
        Solution vector x.

    Algorithm:
        x[n-1] = y[n-1] / U[n-1, n-1]
        x[i] = (y[i] - sum(U[i,j] * x[j] for j > i)) / U[i,i]
    """
    n = len(y)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U.data[i][j] * x[j] for j in range(i + 1, n))) / U.data[i][i]
    return x


def solve_lu(L: Matrix, U: Matrix, b: list[float], P: Matrix = None) -> list[float]:
    """
    Solve Ax = b using precomputed LU decomposition.

    If P is provided, solves PAx = Pb, i.e., LUx = Pb.

    Args:
        L: Lower triangular matrix.
        U: Upper triangular matrix.
        b: Right-hand side vector.
        P: Optional permutation matrix.

    Returns:
        Solution vector x.

    Algorithm:
        1. If P given, compute Pb
        2. Solve Ly = Pb (forward substitution)
        3. Solve Ux = y (backward substitution)
    """
    # Apply permutation if provided
    if P is not None:
        # Pb = P @ b: permute the right-hand side
        b_vec = Vector(b)
        pb = P @ b_vec
        b_permuted = pb.components
    else:
        b_permuted = b

    # Forward substitution: solve Ly = Pb
    y = forward_substitution(L, b_permuted)

    # Backward substitution: solve Ux = y
    x = backward_substitution(U, y)

    return x


def solve_multiple_systems(A: Matrix, bs: list[list[float]]) -> list[list[float]]:
    """
    Solve multiple systems Ax = b with same A.

    Args:
        A: Coefficient matrix.
        bs: List of right-hand side vectors.

    Returns:
        List of solution vectors.

    This demonstrates the efficiency of LU:
    - Factor once
    - Solve many times
    """
    # Compute PLU decomposition once
    P, L, U = plu_decomposition(A)

    # Solve for each b using the decomposition
    solutions = []
    for b in bs:
        x = solve_lu(L, U, b, P)
        solutions.append(x)

    return solutions
