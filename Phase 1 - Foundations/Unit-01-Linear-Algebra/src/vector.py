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

def linear_combination(vectors: List[Vector], scalars: List[float]) -> Vector:
    """
    Compute a linear combination of vectors.

    Args:
        vectors: List of vectors [v₁, v₂, ..., vₙ]
        scalars: List of scalars [c₁, c₂, ..., cₙ]

    Returns:
        Vector: c₁v₁ + c₂v₂ + ... + cₙvₙ

    Raises:
        ValueError: If lengths don't match or vectors have different dimensions.

    Example:
        >>> v1 = Vector([1, 0])
        >>> v2 = Vector([0, 1])
        >>> linear_combination([v1, v2], [3, 4])
        Vector([3, 4])
    """
    if len(vectors) != len(scalars):
        raise ValueError("Different amount of vectors and scalars")
    for vec in vectors:
        if len(vec.components) != len(vectors[0].components):
            raise ValueError("Dimensions don't match")
    zeros = []
    for i in range(len(vectors[0].components)):
        zeros.append(0)
    toReturn = Vector(zeros)
    for vec, scalar in zip(vectors, scalars):
        toReturn = toReturn + vec * scalar
    
    return toReturn
    
def are_linearly_independent(vectors: List[Vector]) -> bool:
    """
    Check if a set of vectors is linearly independent.

    For 2D with 2 vectors, they're independent if not parallel.
    For general case, use determinant or row reduction (advanced).

    Args:
        vectors: List of vectors to check.

    Returns:
        True if linearly independent, False otherwise.

    Simplified approach for 2 vectors in 2D:
        Independent if v1 × v2 ≠ 0 (cross product / determinant)
        For [a,b] and [c,d]: ad - bc ≠ 0
    """
    # TODO: Implement for 2D case first
    # TODO: Check if vectors are parallel (one is scalar multiple of other)
    # Hint: For 2D, compute ad - bc and check if near zero
    if len(vectors) != 2:
        raise ValueError("Only 2 vector lists are supported")
    for vec in vectors:
        if len(vec.components) != 2:
            raise ValueError("Only 2D vectors are supported")
    a,b = vectors[0].components
    c,d = vectors[1].components
    return (a*d-b*c) != 0

def project_onto(v: Vector, onto: Vector) -> Vector:
    """
    Project vector v onto vector 'onto'.

    The projection is the component of v in the direction of 'onto'.

    Args:
        v: Vector to project.
        onto: Vector to project onto.

    Returns:
        Projection of v onto 'onto'.

    Formula:
        proj = ((v · onto) / (onto · onto)) * onto

    Geometric meaning:
        Drop a perpendicular from tip of v to the line of 'onto'.
        Projection is from origin to where perpendicular hits.
    """
    if onto.magnitude() == 0:
        raise ValueError("Cannot project onto zero vector")
    proj_v = (v.dot(onto) / (onto.dot(onto))) * onto
    return proj_v
    
def component_orthogonal_to(v: Vector, onto: Vector) -> Vector:
    """
    Get the component of v perpendicular to 'onto'.

    Args:
        v: Original vector.
        onto: Vector defining the direction.

    Returns:
        Component of v perpendicular to 'onto'.

    Formula:
        v_perp = v - proj(v onto 'onto')

    This + projection = original vector.
    """
    proj_v = project_onto(v, onto)
    return v - proj_v
    
        
        
            
        
            