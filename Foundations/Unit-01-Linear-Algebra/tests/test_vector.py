import pytest
import math
import sys
sys.path.append("..")
from src.vector import Vector

"""
Unit tests for Vector class.
Write these FIRST, then implement to make them pass.
"""


class TestVectorCreation:
    """Test vector instantiation."""

    def test_create_2d_vector(self):
        v = Vector([3, 4])
        assert v.components == [3, 4]

    def test_create_3d_vector(self):
        v = Vector([1, 2, 3])
        assert v.components == [1, 2, 3]

    def test_dimension_property(self):
        v = Vector([1, 2, 3, 4])
        assert v.dimension == 4

    def test_string_representation(self):
        v = Vector([3, 4])
        assert str(v) == "Vector([3, 4])"


class TestVectorAddition:
    """Test vector addition."""

    def test_add_2d_vectors(self):
        v1 = Vector([1, 2])
        v2 = Vector([3, 4])
        result = v1 + v2
        assert result.components == [4, 6]

    def test_add_3d_vectors(self):
        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])
        result = v1 + v2
        assert result.components == [5, 7, 9]

    def test_add_dimension_mismatch_raises(self):
        v1 = Vector([1, 2])
        v2 = Vector([1, 2, 3])
        with pytest.raises(ValueError):
            v1 + v2

    def test_addition_is_commutative(self):
        v1 = Vector([1, 2])
        v2 = Vector([3, 4])
        assert (v1 + v2).components == (v2 + v1).components


class TestScalarMultiplication:
    """Test scalar multiplication."""

    def test_multiply_by_positive_scalar(self):
        v = Vector([2, 3])
        result = v * 2
        assert result.components == [4, 6]

    def test_multiply_by_negative_scalar(self):
        v = Vector([2, 3])
        result = v * -1
        assert result.components == [-2, -3]

    def test_multiply_by_zero(self):
        v = Vector([2, 3])
        result = v * 0
        assert result.components == [0, 0]

    def test_multiply_by_fraction(self):
        v = Vector([4, 6])
        result = v * 0.5
        assert result.components == [2.0, 3.0]

    def test_rmul_scalar_first(self):
        """Test 2 * v works same as v * 2."""
        v = Vector([2, 3])
        result = 2 * v
        assert result.components == [4, 6]


class TestMagnitude:
    """Test vector magnitude (length)."""

    def test_magnitude_3_4_5_triangle(self):
        v = Vector([3, 4])
        assert v.magnitude() == 5.0

    def test_magnitude_unit_vector(self):
        v = Vector([1, 0])
        assert v.magnitude() == 1.0

    def test_magnitude_3d(self):
        v = Vector([1, 2, 2])
        assert v.magnitude() == 3.0

    def test_magnitude_zero_vector(self):
        v = Vector([0, 0])
        assert v.magnitude() == 0.0


class TestDotProduct:
    """Test dot product."""

    def test_dot_product_basic(self):
        v1 = Vector([1, 2])
        v2 = Vector([3, 4])
        # 1*3 + 2*4 = 3 + 8 = 11
        assert v1.dot(v2) == 11

    def test_dot_product_perpendicular(self):
        """Perpendicular vectors have dot product = 0."""
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        assert v1.dot(v2) == 0

    def test_dot_product_parallel(self):
        """Parallel vectors: dot product = product of magnitudes."""
        v1 = Vector([2, 0])
        v2 = Vector([3, 0])
        assert v1.dot(v2) == 6  # |v1| * |v2| = 2 * 3

    def test_dot_product_opposite(self):
        """Opposite vectors: negative dot product."""
        v1 = Vector([1, 0])
        v2 = Vector([-1, 0])
        assert v1.dot(v2) == -1

    def test_dot_dimension_mismatch_raises(self):
        v1 = Vector([1, 2])
        v2 = Vector([1, 2, 3])
        with pytest.raises(ValueError):
            v1.dot(v2)


class TestNormalize:
    """Test vector normalization."""

    def test_normalize_creates_unit_vector(self):
        v = Vector([3, 4])
        normalized = v.normalize()
        assert abs(normalized.magnitude() - 1.0) < 1e-10

    def test_normalize_preserves_direction(self):
        v = Vector([3, 4])
        normalized = v.normalize()
        # Direction preserved: components should be [0.6, 0.8]
        assert abs(normalized.components[0] - 0.6) < 1e-10
        assert abs(normalized.components[1] - 0.8) < 1e-10

    def test_normalize_zero_vector_raises(self):
        v = Vector([0, 0])
        with pytest.raises(ValueError):
            v.normalize()


class TestSubtraction:
    """Test vector subtraction."""

    def test_subtract_vectors(self):
        v1 = Vector([5, 7])
        v2 = Vector([2, 3])
        result = v1 - v2
        assert result.components == [3, 4]

    def test_subtract_gives_displacement(self):
        """v2 - v1 gives vector FROM v1 TO v2."""
        point1 = Vector([1, 1])
        point2 = Vector([4, 5])
        displacement = point2 - point1
        assert displacement.components == [3, 4]


class TestEquality:
    """Test vector equality."""

    def test_equal_vectors(self):
        v1 = Vector([1, 2])
        v2 = Vector([1, 2])
        assert v1 == v2

    def test_unequal_vectors(self):
        v1 = Vector([1, 2])
        v2 = Vector([1, 3])
        assert v1 != v2

    def test_different_dimensions_not_equal(self):
        v1 = Vector([1, 2])
        v2 = Vector([1, 2, 3])
        assert v1 != v2