"""
Matrix class implementation from scratch
Matrices as collections of column vectors
"""

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
    
        