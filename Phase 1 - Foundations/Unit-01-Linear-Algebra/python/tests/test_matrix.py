"""Tests for Matrix class."""
import pytest
from src.matrix import (Matrix, 
    rotation_matrix,
    shear_matrix,
    scaling_matrix,
    reflection_matrix,
    projection_matrix,
    compose)
from src.vector import Vector


class TestMatrixCreation:
    def test_create_2x2(self):
        m = Matrix([[1, 2], [3, 4]])
        assert m.shape == (2, 2)

    def test_create_2x3(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        assert m.shape == (2, 3)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            Matrix([])

    def test_uneven_rows_raises(self):
        with pytest.raises(ValueError):
            Matrix([[1, 2], [3]])

    def test_element_access(self):
        m = Matrix([[1, 2], [3, 4]])
        assert m[0, 0] == 1
        assert m[0, 1] == 2
        assert m[1, 0] == 3
        assert m[1, 1] == 4

    def test_element_set(self):
        m = Matrix([[1, 2], [3, 4]])
        m[0, 0] = 99
        assert m[0, 0] == 99


class TestMatrixRowColumn:
    def test_get_row(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        row = m.get_row(0)
        assert row.components == [1, 2, 3]

    def test_get_column(self):
        m = Matrix([[1, 2], [3, 4], [5, 6]])
        col = m.get_column(0)
        assert col.components == [1, 3, 5]


class TestMatrixAddition:
    def test_add_2x2(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        result = m1 + m2
        assert result[0, 0] == 6
        assert result[1, 1] == 12

    def test_add_shape_mismatch_raises(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(ValueError):
            m1 + m2


class TestScalarMultiplication:
    def test_multiply_by_scalar(self):
        m = Matrix([[1, 2], [3, 4]])
        result = m * 2
        assert result[0, 0] == 2
        assert result[1, 1] == 8

    def test_rmul(self):
        m = Matrix([[1, 2], [3, 4]])
        result = 2 * m
        assert result[0, 0] == 2


class TestMatrixVectorMultiplication:
    def test_2x2_times_2d_vector(self):
        m = Matrix([[1, 2], [3, 4]])
        v = Vector([5, 6])
        result = m @ v
        # [1*5+2*6, 3*5+4*6] = [17, 39]
        assert result.components == [17, 39]

    def test_identity_times_vector(self):
        I = Matrix.identity(2)
        v = Vector([3, 4])
        result = I @ v
        assert result.components == [3, 4]

    def test_dimension_mismatch_raises(self):
        m = Matrix([[1, 2], [3, 4]])
        v = Vector([1, 2, 3])
        with pytest.raises(ValueError):
            m @ v


class TestMatrixMatrixMultiplication:
    def test_2x2_times_2x2(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])
        result = m1 @ m2
        # [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
        # = [[19, 22], [43, 50]]
        assert result[0, 0] == 19
        assert result[0, 1] == 22
        assert result[1, 0] == 43
        assert result[1, 1] == 50

    def test_2x3_times_3x2(self):
        m1 = Matrix([[1, 2, 3], [4, 5, 6]])
        m2 = Matrix([[7, 8], [9, 10], [11, 12]])
        result = m1 @ m2
        assert result.shape == (2, 2)

    def test_identity_multiplication(self):
        m = Matrix([[1, 2], [3, 4]])
        I = Matrix.identity(2)
        assert (m @ I)[0, 0] == m[0, 0]
        assert (I @ m)[0, 0] == m[0, 0]

    def test_dimension_mismatch_raises(self):
        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        with pytest.raises(ValueError):
            m1 @ m2


class TestTranspose:
    def test_transpose_2x2(self):
        m = Matrix([[1, 2], [3, 4]])
        t = m.transpose()
        assert t[0, 0] == 1
        assert t[0, 1] == 3
        assert t[1, 0] == 2
        assert t[1, 1] == 4

    def test_transpose_2x3(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])
        t = m.transpose()
        assert t.shape == (3, 2)

    def test_double_transpose(self):
        m = Matrix([[1, 2], [3, 4]])
        assert m.transpose().transpose() == m


class TestSpecialMatrices:
    def test_identity(self):
        I = Matrix.identity(3)
        assert I.shape == (3, 3)
        assert I[0, 0] == 1
        assert I[1, 1] == 1
        assert I[0, 1] == 0

    def test_zeros(self):
        z = Matrix.zeros(2, 3)
        assert z.shape == (2, 3)
        assert z[0, 0] == 0
        assert z[1, 2] == 0

    def test_from_column_vectors(self):
        v1 = Vector([1, 2])
        v2 = Vector([3, 4])
        m = Matrix.from_column_vectors([v1, v2])
        assert m.shape == (2, 2)
        assert m.get_column(0).components == [1, 2]
        assert m.get_column(1).components == [3, 4]
    
    class TestRotation:
        def test_rotate_90_degrees(self):
            R = rotation_matrix(90)
            v = Vector([1, 0])
            result = R @ v
            # [1,0] rotated 90° CCW should be [0,1]
            assert abs(result.components[0] - 0) < 1e-10
            assert abs(result.components[1] - 1) < 1e-10
    
        def test_rotate_180_degrees(self):
            R = rotation_matrix(180)
            v = Vector([1, 0])
            result = R @ v
            # Should be [-1, 0]
            assert abs(result.components[0] - (-1)) < 1e-10
            assert abs(result.components[1] - 0) < 1e-10
    
        def test_rotate_360_returns_original(self):
            R = rotation_matrix(360)
            v = Vector([3, 4])
            result = R @ v
            assert abs(result.components[0] - 3) < 1e-10
            assert abs(result.components[1] - 4) < 1e-10
    
        def test_rotation_preserves_magnitude(self):
            R = rotation_matrix(45)
            v = Vector([3, 4])
            result = R @ v
            assert abs(result.magnitude() - v.magnitude()) < 1e-10
        
        
    class TestScaling:
        def test_scale_by_2(self):
            S = scaling_matrix(2, 2)
            v = Vector([3, 4])
            result = S @ v
            assert result.components == [6, 8]
    
        def test_scale_non_uniform(self):
            S = scaling_matrix(2, 3)
            v = Vector([1, 1])
            result = S @ v
            assert result.components == [2, 3]
    
        def test_scale_by_half(self):
            S = scaling_matrix(0.5, 0.5)
            v = Vector([4, 6])
            result = S @ v
            assert result.components == [2, 3]
    
    
    class TestShear:
        def test_horizontal_shear(self):
            Sh = shear_matrix(kx=1)
            v = Vector([0, 1])
            result = Sh @ v
            # [0,1] with kx=1 becomes [1, 1]
            assert result.components == [1, 1]
    
        def test_no_shear(self):
            Sh = shear_matrix(kx=0, ky=0)
            v = Vector([3, 4])
            result = Sh @ v
            assert result.components == [3, 4]
    
    
    class TestReflection:
        def test_reflect_x_axis(self):
            R = reflection_matrix('x')
            v = Vector([3, 4])
            result = R @ v
            assert result.components == [3, -4]
    
        def test_reflect_y_axis(self):
            R = reflection_matrix('y')
            v = Vector([3, 4])
            result = R @ v
            assert result.components == [-3, 4]
    
    
    class TestProjection:
        def test_project_onto_x(self):
            P = projection_matrix('x')
            v = Vector([3, 4])
            result = P @ v
            assert result.components == [3, 0]
    
        def test_project_onto_y(self):
            P = projection_matrix('y')
            v = Vector([3, 4])
            result = P @ v
            assert result.components == [0, 4]
    
    
    class TestComposition:
        def test_compose_two_transformations(self):
            S = scaling_matrix(2, 2)
            R = rotation_matrix(90)
            # Scale by 2, then rotate 90
            combined = compose(S, R)
            v = Vector([1, 0])
            # Scale: [1,0] -> [2,0]
            # Rotate: [2,0] -> [0,2]
            result = combined @ v
            assert abs(result.components[0] - 0) < 1e-10
            assert abs(result.components[1] - 2) < 1e-10
    
        def test_rotation_inverse(self):
            R = rotation_matrix(45)
            R_inv = rotation_matrix(-45)
            combined = compose(R, R_inv)
            # Should be identity
            v = Vector([3, 4])
            result = combined @ v
            assert abs(result.components[0] - 3) < 1e-10
            assert abs(result.components[1] - 4) < 1e-10