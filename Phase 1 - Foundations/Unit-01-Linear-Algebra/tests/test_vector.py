import pytest
import math
from src.vector import (
Vector,
linear_combination,
are_linearly_independent,
project_onto,
component_orthogonal_to
)

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

class TestLinearCombination:
    """Test linear combination function."""

    def test_basic_combination(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        result = linear_combination([v1, v2], [3, 4])
        assert result.components == [3, 4]

    def test_single_vector(self):
        v = Vector([2, 3])
        result = linear_combination([v], [2])
        assert result.components == [4, 6]

    def test_zero_scalars(self):
        v1 = Vector([1, 2])
        v2 = Vector([3, 4])
        result = linear_combination([v1, v2], [0, 0])
        assert result.components == [0, 0]

    def test_negative_scalars(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        result = linear_combination([v1, v2], [-1, 2])
        assert result.components == [-1, 2]

    def test_mismatched_lengths_raises(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        with pytest.raises(ValueError):
            linear_combination([v1, v2], [1])  # Only one scalar


class TestLinearIndependence:
    """Test linear independence check."""

    def test_independent_standard_basis(self):
        v1 = Vector([1, 0])
        v2 = Vector([0, 1])
        assert are_linearly_independent([v1, v2]) 

    def test_dependent_parallel_vectors(self):
        v1 = Vector([1, 2])
        v2 = Vector([2, 4])  # 2 * v1
        assert not are_linearly_independent([v1, v2]) 

    def test_dependent_opposite_vectors(self):
        v1 = Vector([1, 2])
        v2 = Vector([-1, -2])  # -1 * v1
        assert not are_linearly_independent([v1, v2]) 

    def test_independent_non_parallel(self):
        v1 = Vector([1, 1])
        v2 = Vector([1, -1])
        assert are_linearly_independent([v1, v2])


class TestProjection:
    """Test vector projection."""

    def test_project_onto_x_axis(self):
        v = Vector([3, 4])
        x_axis = Vector([1, 0])
        proj = project_onto(v, x_axis)
        assert proj.components == [3, 0]

    def test_project_onto_y_axis(self):
        v = Vector([3, 4])
        y_axis = Vector([0, 1])
        proj = project_onto(v, y_axis)
        assert proj.components == [0, 4]

    def test_project_parallel_vectors(self):
        v = Vector([3, 4])
        onto = Vector([3, 4])
        proj = project_onto(v, onto)
        # Projecting onto itself should give itself
        assert abs(proj.components[0] - 3) < 1e-10
        assert abs(proj.components[1] - 4) < 1e-10

    def test_project_perpendicular_vectors(self):
        v = Vector([1, 0])
        onto = Vector([0, 1])
        proj = project_onto(v, onto)
        # Perpendicular projection should be zero
        assert abs(proj.components[0]) < 1e-10
        assert abs(proj.components[1]) < 1e-10


class TestOrthogonalComponent:
    """Test orthogonal component."""

    def test_orthogonal_to_x_axis(self):
        v = Vector([3, 4])
        x_axis = Vector([1, 0])
        perp = component_orthogonal_to(v, x_axis)
        assert perp.components == [0, 4]

    def test_projection_plus_orthogonal_equals_original(self):
        v = Vector([3, 4])
        onto = Vector([1, 1])
        proj = project_onto(v, onto)
        perp = component_orthogonal_to(v, onto)
        reconstructed = proj + perp
        assert abs(reconstructed.components[0] - v.components[0]) < 1e-10
        assert abs(reconstructed.components[1] - v.components[1]) < 1e-10

    def test_orthogonal_is_perpendicular(self):
        v = Vector([3, 4])
        onto = Vector([1, 1])
        perp = component_orthogonal_to(v, onto)
        # Perpendicular component should have zero dot product with 'onto'
        assert abs(perp.dot(onto)) < 1e-10