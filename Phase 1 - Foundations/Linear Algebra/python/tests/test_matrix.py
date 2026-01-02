"""Tests for Matrix class."""
import pytest
from src.matrix import (Matrix, 
    rotation_matrix,
    shear_matrix,
    scaling_matrix,
    reflection_matrix,
    projection_matrix,
    compose, determinant,
 determinant_2x2, determinant_3x3, 
 is_invertible, minor, cofactor,
 cofactor_matrix, adjugate,
 inverse, inverse_2x2, inverse_adjugate,
 inverse_gauss_jordan, swap_rows, scale_row,
 add_row_multiple, augment,forward_elimination,
 back_substitution, solve_system, gaussian_elimination,
 solve_with_inverse, rank, is_consistent
)
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
    class TestDeterminant2x2:
        def test_identity(self):
            I = Matrix([[1, 0], [0, 1]])
            assert determinant_2x2(I) == 1
    
        def test_basic(self):
            A = Matrix([[3, 8], [4, 6]])
            # det = 3*6 - 8*4 = 18 - 32 = -14
            assert determinant_2x2(A) == -14
    
        def test_singular(self):
            # Rows are multiples of each other
            A = Matrix([[2, 4], [1, 2]])
            assert determinant_2x2(A) == 0
    
        def test_rotation_matrix(self):
            import math
            theta = math.pi / 4  # 45 degrees
            R = Matrix([
                [math.cos(theta), -math.sin(theta)],
                [math.sin(theta), math.cos(theta)]
            ])
            # Rotation matrices have det = 1
            assert abs(determinant_2x2(R) - 1) < 1e-10
    
    
    class TestDeterminant3x3:
        def test_identity(self):
            I = Matrix([[1,0,0], [0,1,0], [0,0,1]])
            assert determinant_3x3(I) == 1
    
        def test_basic(self):
            A = Matrix([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ])
            # This matrix is singular (det = 0)
            assert abs(determinant_3x3(A)) < 1e-10
    
        def test_nonsingular(self):
            A = Matrix([
                [1, 2, 3],
                [0, 1, 4],
                [5, 6, 0]
            ])
            # det = 1(0-24) - 2(0-20) + 3(0-5) = -24 + 40 - 15 = 1
            assert determinant_3x3(A) == 1
    
    
    class TestMinor:
        def test_3x3_minor(self):
            A = Matrix([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ])
            M = minor(A, 0, 0)
            assert M.data == [[5, 6], [8, 9]]
    
            M = minor(A, 1, 1)
            assert M.data == [[1, 3], [7, 9]]
    
    
    class TestCofactor:
        def test_cofactor_signs(self):
            A = Matrix([
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ])
            # Cofactor at (0,0): +1 * det([[5,6],[8,9]]) = 5*9 - 6*8 = -3
            assert cofactor(A, 0, 0) == -3
    
            # Cofactor at (0,1): -1 * det([[4,6],[7,9]]) = -(4*9 - 6*7) = -(-6) = 6
            assert cofactor(A, 0, 1) == 6
    
    
    class TestDeterminantGeneral:
        def test_1x1(self):
            A = Matrix([[5]])
            assert determinant(A) == 5
    
        def test_4x4(self):
            A = Matrix([
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
                [13, 14, 15, 16]
            ])
            # This is singular
            assert abs(determinant(A)) < 1e-10
    
        def test_4x4_nonsingular(self):
            A = Matrix([
                [1, 0, 2, -1],
                [3, 0, 0, 5],
                [2, 1, 4, -3],
                [1, 0, 5, 0]
            ])
            # det = 30 (computed separately)
            assert abs(determinant(A) - 30) < 1e-10
    
    
    class TestIsInvertible:
        def test_invertible(self):
            A = Matrix([[1, 2], [3, 4]])
            assert is_invertible(A) == True
    
        def test_singular(self):
            A = Matrix([[1, 2], [2, 4]])
            assert is_invertible(A) == False
class TestInverse2x2:
    def test_identity_inverse(self):
        I = Matrix([[1, 0], [0, 1]])
        I_inv = inverse_2x2(I)
        assert I_inv.data == [[1, 0], [0, 1]]

    def test_basic_inverse(self):
        A = Matrix([[4, 7], [2, 6]])
        A_inv = inverse_2x2(A)
        # det = 24 - 14 = 10
        # A_inv = (1/10) * [[6, -7], [-2, 4]]
        assert abs(A_inv.data[0][0] - 0.6) < 1e-10
        assert abs(A_inv.data[0][1] - (-0.7)) < 1e-10
        assert abs(A_inv.data[1][0] - (-0.2)) < 1e-10
        assert abs(A_inv.data[1][1] - 0.4) < 1e-10

    def test_inverse_times_original(self):
        A = Matrix([[4, 7], [2, 6]])
        A_inv = inverse_2x2(A)
        product = A @ A_inv
        # Should be identity
        assert abs(product.data[0][0] - 1) < 1e-10
        assert abs(product.data[0][1] - 0) < 1e-10
        assert abs(product.data[1][0] - 0) < 1e-10
        assert abs(product.data[1][1] - 1) < 1e-10

    def test_singular_matrix_raises(self):
        A = Matrix([[1, 2], [2, 4]])  # det = 0
        with pytest.raises(ValueError):
            inverse_2x2(A)


class TestCofactorMatrix:
    def test_2x2_cofactor_matrix(self):
        A = Matrix([[1, 2], [3, 4]])
        C = cofactor_matrix(A)
        assert C.data == [[4, -3], [-2, 1]]


class TestAdjugate:
    def test_2x2_adjugate(self):
        A = Matrix([[1, 2], [3, 4]])
        adj = adjugate(A)
        # Transpose of cofactor matrix
        assert adj.data == [[4, -2], [-3, 1]]


class TestGaussJordan:
    def test_3x3_inverse(self):
        A = Matrix([
            [1, 2, 3],
            [0, 1, 4],
            [5, 6, 0]
        ])
        A_inv = inverse_gauss_jordan(A)

        # Verify A @ A_inv = I
        product = A @ A_inv
        for i in range(3):
            for j in range(3):
                expected = 1 if i == j else 0
                assert abs(product.data[i][j] - expected) < 1e-10

    def test_4x4_inverse(self):
        A = Matrix([
            [1, 0, 2, -1],
            [3, 0, 0, 5],
            [2, 1, 4, -3],
            [1, 0, 5, 0]
        ])
        A_inv = inverse_gauss_jordan(A)

        # Verify A @ A_inv = I
        product = A @ A_inv
        for i in range(4):
            for j in range(4):
                expected = 1 if i == j else 0
                assert abs(product.data[i][j] - expected) < 1e-10

    def test_singular_raises(self):
        A = Matrix([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ])
        with pytest.raises(ValueError):
            inverse_gauss_jordan(A)


class TestRowOperations:
    def test_swap_rows(self):
        A = Matrix([[1, 2], [3, 4]])
        B = swap_rows(A, 0, 1)
        assert B.data == [[3, 4], [1, 2]]
        # Original unchanged
        assert A.data == [[1, 2], [3, 4]]

    def test_scale_row(self):
        A = Matrix([[1, 2], [3, 4]])
        B = scale_row(A, 0, 2)
        assert B.data == [[2, 4], [3, 4]]

    def test_add_row_multiple(self):
        A = Matrix([[1, 2], [3, 4]])
        # row[1] += 2 * row[0]
        B = add_row_multiple(A, 1, 0, 2)
        assert B.data == [[1, 2], [5, 8]]


class TestInverseDispatcher:
    def test_uses_2x2_formula(self):
        A = Matrix([[4, 7], [2, 6]])
        A_inv = inverse(A)
        # Check it worked
        product = A @ A_inv
        assert abs(product.data[0][0] - 1) < 1e-10

    def test_uses_gauss_jordan_for_larger(self):
        A = Matrix([
            [1, 2, 3],
            [0, 1, 4],
            [5, 6, 0]
        ])
        A_inv = inverse(A)
        product = A @ A_inv
        for i in range(3):
            for j in range(3):
                expected = 1 if i == j else 0
                assert abs(product.data[i][j] - expected) < 1e-10

class TestForwardElimination:
    def test_simple_2x2(self):
        # 2x + 3y = 8
        # x - 2y = -3
        augmented = Matrix([
            [2, 3, 8],
            [1, -2, -3]
        ])
        ref, rank = forward_elimination(augmented)
        assert rank == 2  # Full rank

    def test_rank_deficient(self):
        # Rows are dependent
        augmented = Matrix([
            [1, 2, 3, 6],
            [2, 4, 6, 12],
            [1, 1, 1, 3]
        ])
        ref, rank = forward_elimination(augmented)
        assert rank == 2  # Only 2 independent rows


class TestBackSubstitution:
    def test_simple_solution(self):
        # Already in row echelon form
        ref = Matrix([
            [1, 2, 5],   # x + 2y = 5
            [0, 1, 2]    # y = 2
        ])
        solution = back_substitution(ref, 2)
        assert solution is not None
        assert abs(solution[0] - 1) < 1e-10  # x = 1
        assert abs(solution[1] - 2) < 1e-10  # y = 2


class TestSolveSystem:
    def test_unique_solution(self):
        A = Matrix([[2, 3], [1, -2]])
        b = [8, -3]
        result = solve_system(A, b)
        assert result['status'] == 'unique'
        assert abs(result['solution'][0] - 1) < 1e-10  # x = 1
        assert abs(result['solution'][1] - 2) < 1e-10  # y = 2

    def test_inconsistent_system(self):
        # Parallel lines
        A = Matrix([[1, 1], [1, 1]])
        b = [2, 3]  # Can't both be true
        result = solve_system(A, b)
        assert result['status'] == 'inconsistent'

    def test_infinite_solutions(self):
        # Same line twice
        A = Matrix([[1, 1], [2, 2]])
        b = [2, 4]  # Second is 2x first
        result = solve_system(A, b)
        assert result['status'] == 'infinite'

    def test_3x3_system(self):
        A = Matrix([
            [1, 1, 1],
            [0, 2, 5],
            [2, 5, -1]
        ])
        b = [6, -4, 27]
        result = solve_system(A, b)
        assert result['status'] == 'unique'
        # x=5, y=3, z=-2
        x = result['solution']
        assert abs(x[0] - 5) < 1e-10
        assert abs(x[1] - 3) < 1e-10
        assert abs(x[2] - (-2)) < 1e-10


class TestGaussianElimination:
    def test_basic(self):
        A = Matrix([[2, 3], [1, -2]])
        b = [8, -3]
        x = gaussian_elimination(A, b)
        assert abs(x[0] - 1) < 1e-10
        assert abs(x[1] - 2) < 1e-10

    def test_raises_on_singular(self):
        A = Matrix([[1, 1], [1, 1]])
        b = [2, 3]
        with pytest.raises(ValueError):
            gaussian_elimination(A, b)


class TestSolveWithInverse:
    def test_matches_gaussian(self):
        A = Matrix([[2, 3], [1, -2]])
        b = [8, -3]
        x_gauss = gaussian_elimination(A, b)
        x_inv = solve_with_inverse(A, b)
        for i in range(len(x_gauss)):
            assert abs(x_gauss[i] - x_inv[i]) < 1e-10


class TestRank:
    def test_full_rank(self):
        A = Matrix([[1, 2], [3, 4]])
        assert rank(A) == 2

    def test_rank_deficient(self):
        A = Matrix([[1, 2], [2, 4]])
        assert rank(A) == 1

    def test_rectangular(self):
        A = Matrix([
            [1, 2, 3],
            [4, 5, 6]
        ])
        assert rank(A) == 2  # At most min(rows, cols)


class TestConsistency:
    def test_consistent(self):
        A = Matrix([[1, 2], [3, 4]])
        b = [5, 6]
        assert is_consistent(A, b) == True

    def test_inconsistent(self):
        A = Matrix([[1, 1], [1, 1]])
        b = [2, 3]
        assert is_consistent(A, b) == False
