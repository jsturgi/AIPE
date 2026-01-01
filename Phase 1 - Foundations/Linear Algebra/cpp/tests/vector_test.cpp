#include <gtest/gtest.h>
#include "vector.hpp"

// Test construction and dimension
TEST(VectorTest, ConstructorAndDimension) {
    Vector<3> v = {1.0, 2.0, 3.0};
    EXPECT_EQ(v.dimension(), 3);
    EXPECT_DOUBLE_EQ(v[0], 1.0);
    EXPECT_DOUBLE_EQ(v[1], 2.0);
    EXPECT_DOUBLE_EQ(v[2], 3.0);
}

TEST(VectorTest, DefaultConstructorZeroVector) {
    Vector<3> v;
    EXPECT_DOUBLE_EQ(v[0], 0.0);
    EXPECT_DOUBLE_EQ(v[1], 0.0);
    EXPECT_DOUBLE_EQ(v[2], 0.0);
}

// Test addition
TEST(VectorTest, Addition) {
    Vector<3> v1 = {1.0, 2.0, 3.0};
    Vector<3> v2 = {4.0, 5.0, 6.0};
    Vector<3> result = v1 + v2;

    EXPECT_DOUBLE_EQ(result[0], 5.0);
    EXPECT_DOUBLE_EQ(result[1], 7.0);
    EXPECT_DOUBLE_EQ(result[2], 9.0);
}

// Test subtraction
TEST(VectorTest, Subtraction) {
    Vector<3> v1 = {4.0, 5.0, 6.0};
    Vector<3> v2 = {1.0, 2.0, 3.0};
    Vector<3> result = v1 - v2;

    EXPECT_DOUBLE_EQ(result[0], 3.0);
    EXPECT_DOUBLE_EQ(result[1], 3.0);
    EXPECT_DOUBLE_EQ(result[2], 3.0);
}

// Test scalar multiplication
TEST(VectorTest, ScalarMultiplication) {
    Vector<3> v = {1.0, 2.0, 3.0};
    Vector<3> result = v * 2.0;

    EXPECT_DOUBLE_EQ(result[0], 2.0);
    EXPECT_DOUBLE_EQ(result[1], 4.0);
    EXPECT_DOUBLE_EQ(result[2], 6.0);
}

// Test dot product
TEST(VectorTest, DotProduct) {
    Vector<3> v1 = {1.0, 2.0, 3.0};
    Vector<3> v2 = {4.0, 5.0, 6.0};
    double result = v1.dot(v2);

    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_DOUBLE_EQ(result, 32.0);
}

// Test magnitude
TEST(VectorTest, Magnitude) {
    Vector<3> v = {3.0, 4.0, 0.0};
    // sqrt(9 + 16 + 0) = sqrt(25) = 5
    EXPECT_DOUBLE_EQ(v.magnitude(), 5.0);
}

TEST(VectorTest, MagnitudeUnitVector) {
    Vector<3> v = {1.0, 0.0, 0.0};
    EXPECT_DOUBLE_EQ(v.magnitude(), 1.0);
}

// Test normalize
TEST(VectorTest, Normalize) {
    Vector<3> v = {3.0, 4.0, 0.0};
    Vector<3> unit = v.normalize();

    EXPECT_NEAR(unit[0], 0.6, 1e-10);
    EXPECT_NEAR(unit[1], 0.8, 1e-10);
    EXPECT_NEAR(unit[2], 0.0, 1e-10);
    EXPECT_NEAR(unit.magnitude(), 1.0, 1e-10);
}

TEST(VectorTest, NormalizeZeroVectorThrows) {
    Vector<3> zero;
    EXPECT_THROW(zero.normalize(), std::runtime_error);
}

// Test approx_equal
TEST(VectorTest, ApproxEqualTrue) {
    Vector<3> v1 = {1.0, 2.0, 3.0};
    Vector<3> v2 = {1.0 + 1e-11, 2.0, 3.0};
    EXPECT_TRUE(v1.approx_equal(v2));
}

TEST(VectorTest, ApproxEqualFalse) {
    Vector<3> v1 = {1.0, 2.0, 3.0};
    Vector<3> v2 = {1.1, 2.0, 3.0};
    EXPECT_FALSE(v1.approx_equal(v2));
}

// Test with different dimensions
TEST(VectorTest, TwoDimensional) {
    Vector<2> v1 = {3.0, 4.0};
    EXPECT_EQ(v1.dimension(), 2);
    EXPECT_DOUBLE_EQ(v1.magnitude(), 5.0);
}
