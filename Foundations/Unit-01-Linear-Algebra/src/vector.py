"""
Vector class implementation from scratch
"""
import math
from typing import List, Union

class Vector:
    """
    A mathematical vector with geometrical operations
    
    Vectors are arrows with direction and magnitude.
    This class implements vectors as lists of components.
    
    Attributes:
        components: List of numerical values [x, y, z,...]
    
    Example:
        >>> v = Vector([3,4])
        >>> v.magnitude()
        5.0
    """
    
    def __init__(self, components: List[Union[int, float]]):
        """
        Initialize a vector with given components.
        
        Args:
            components: List of numbers representing vector components
        
        Raises:
            ValueError: If components is empty
        """
        self.components = list(components)
        if not self.components:
            raise ValueError("Vector components cannot be empty")
    
    @property
    def dimension(self) -> int:
        """
        Return the dimension (number of components) of the vector.
        
        Returns:
            Number of components
        """
        return len(self.components)
    
    def __repr__(self) -> str:
        """
        String representation of the vector.
        
        Returns:
            "Vector([4,3])
        """
        return f"Vector({self.components})"
    
    def __eq__(self, other: object) -> bool:
        """
        Check if two vectors are equal.
        
        Args:
            other: Another vector to compare.
        
        Returns:
            True if components match, False otherwise.
        """
        if type(other) is not Vector:
            raise TypeError("Other must be a Vector")
        return self.components == other.components
    
    def __add__(self, other: 'Vector') -> 'Vector':
        """
        Add two vectors component-wise.
        
        Args:
            other: Vector to add.
        
        Returns:
            New vector that is the sum
        
        Raises:
            ValueError: If dimensions don't match
        
        Geometric Meaning:
            Place other's tail at self's tip.
            Result goes from self's tail to other's tip.
        """
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must be the same dimension")
        result = [a + b for a, b in zip(self.components, other.components)]
        return Vector(result)
    
    def __sub__(self, other: 'Vector') -> 'Vector':
        """
        Subtract other vector from self.
        
        Args:
            other: Vector to subtract.
        
        Returns:
            New vector that is the difference
        
        Geometric Meaning:
            self - other gives vector FROM other TO self.
        """
        if len(self.components) != len(other.components):
                raise ValueError("Vectors are of different dimension!")
        result = [a-b for a,b in zip(self.components, other.components)]
        return Vector(result)
    
    def __mul__(self, scalar: Union[int, float]) -> 'Vector':
        """
        Multiply vector by a scalar
        
        Args:
            scalar: Number to multiply by
        
        Returns:
            New scaled vector.
        
        Geometric meaning:
            Stretches (or shrinks) the arrow.
            Negative scalar reverses direction.
        """
        result = [scalar * component for component in self.components]
        return Vector(result)
    
    def __rmul__(self, scalar: Union[int, float]) -> 'Vector':
        """
        Allow scalar * vector (scalar on left side).

        Args:
            scalar: Number to multiply by.

        Returns:
            New scaled vector.
        """
        return self.__mul__(scalar)

    def magnitude(self) -> float:
        """
        Calculate the magnitude (length) of the vector.

        Returns:
            The magnitude (always non-negative).

        Formula:
            |v| = sqrt(x² + y² + z² + ...)

        Geometric meaning:
            The length of the arrow.
        """
        sum_of_squares = sum(c**2 for c in self.components)
        return sum_of_squares ** 0.5

    def dot(self, other: 'Vector') -> float:
        """
        Compute dot product with another vector.

        Args:
            other: Vector to dot with.

        Returns:
            Scalar dot product value.

        Raises:
            ValueError: If dimensions don't match.

        Formula:
            v1 · v2 = x1*x2 + y1*y2 + z1*z2 + ...

        Geometric meaning:
            - Positive: vectors point similar direction
            - Zero: vectors are perpendicular
            - Negative: vectors point opposite directions

        Also equals:
            |v1| * |v2| * cos(angle between them)
        """
        if len(self.components) != len(other.components):
            raise ValueError("Vectors must be the same dimension")
        return sum(a * b for a, b in zip(self.components, other.components))

    def normalize(self) -> 'Vector':
        """
        Return a unit vector in the same direction.

        Returns:
            New vector with magnitude 1.

        Raises:
            ValueError: If vector has zero magnitude.

        Formula:
            v_normalized = v / |v|

        Geometric meaning:
            Same direction, but length exactly 1.
        """
        mag = self.magnitude()
        if mag == 0:
            raise ValueError("Cannot normalize a zero vector")
        return Vector([component / self.magnitude() for component in self.components])


def angle_between(v1: Vector, v2: Vector) -> float:
    """
    Calculate angle between two vectors in degrees.

    Args:
        v1: First vector.
        v2: Second vector.

    Returns:
        Angle in degrees (0 to 180).

    Formula:
        cos(θ) = (v1 · v2) / (|v1| * |v2|)
        θ = arccos(cos(θ))

    Hints:
        - Use math.acos() for arccos
        - Use math.degrees() to convert radians to degrees
        - Handle numerical precision (clamp to [-1, 1])
    """
    dot = v1.dot(v2)
    dot = dot / (v1.magnitude() * v2.magnitude())
    angle = math.degrees(math.acos(dot))
    return angle
        
        
            
        
            