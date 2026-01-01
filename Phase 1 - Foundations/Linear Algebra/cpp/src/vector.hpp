#pragma once

#include <array>
#include <cmath>
#include <stdexcept>
#include <initializer_list>

/**
 * Template vector class with compile-time dimension.
 * C++ provides maximum performance and SIMD optimization potential.
 */
template<size_t N>
class Vector {
    static_assert(N > 0, "Vector dimension must be positive");

private:
    std::array<double, N> components_;

public:
    // constructor from initializer list
    Vector(std::initializer_list<double> init){
        if (init.size() != N){
            throw std::invalid_argument("Wrong number \
                of components");
        }
        std::copy(init.begin(), init.end(), components_.begin());
    }
    
    // Default constructor (zero vector)
    Vector() : components_{} {}
    
    // Accessors
    constexpr size_t dimension() const {return N;}
    const double& operator[](size_t i) const { return 
        components_[i];}
    double& operator[](size_t i) { return components_[i];}
    
    /**
     * Vector addition.
     *
     * Returns a new vector that is the sum of this vector and other.
     *
     * @param other - The vector to add
     * @returns The sum vector
     */
     Vector<N> operator+(const Vector<N>& other) const {
         // TODO: Create result vector
         // TODO: Loop through indices 0 to N-1
         // TODO: Add corresponding components
         // TODO: Return result
         Vector<N> result;
         for (int i=0; i < N; i++) {
             result[i] = other[i] + components_[i];
         }
         return result;
     }
    /**
    * Vector subtraction.
    *
    * @param other - The vector to subtract
    * @returns The difference vector
    */
    Vector<N> operator-(const Vector<N>& other) const {
        Vector<N> result;
        for (int i = 0; i < N; i++) {
            result[i] = components_[i] - other[i];
        }
        return result;
    }
    /**
     * Scalar multiplication.
     *
     * @param scalar - The scalar to multiply by
     * @returns The scaled vector
     */
    Vector<N> operator*(double scalar) const {
        Vector<N> result;
        for (int i = 0; i < N; i++){
            result[i] = components_[i] * scalar;
        }
        return result;
    }
    
    /**
     * Dot product.
     *
     * Formula: sum of element-wise products
     *
     * @param other - The vector to compute dot product with
     * @returns The dot product (scalar)
     */
    double dot(const Vector<N>& other) const {
        double sum = 0.0;
        for (int i = 0; i < N; i++){
            sum += components_[i] * other[i];
        }
        return sum;
    }
    
    /**
     * Magnitude (Euclidean norm).
     *
     * @returns The magnitude of the vector
     */
    double magnitude() const {
        double vdot = dot(*this);
        return std::sqrt(vdot);
    }
    /**
     * Normalize to unit vector.
     *
     * @returns A unit vector in the same direction
     * @throws std::runtime_error if vector is zero
     */
    Vector<N> normalize() const {
        // TODO: Compute magnitude
        // TODO: Check if magnitude is near zero (< 1e-10)
        // TODO: Divide vector by magnitude
        // TODO: Return result
        Vector<N> result;
        double mag = magnitude();
        if (mag < 1e-10){
            throw std::runtime_error("Can't normalize a zero vector.");
        }
        for (int i = 0; i < N; i++) {
            result[i] = components_[i] / mag;
        }
        return result;
    }
    /**
     * Check approximate equality.
     *
     * @param other - The vector to compare with
     * @param tolerance - Maximum allowed difference
     * @returns True if approximately equal
     */
    bool approx_equal(const Vector<N>& other, double tolerance = 1e-10) const {
        for (int i = 0; i < N; i++) {
            if (std::abs(components_[i] - other[i]) > tolerance) {
                return false;
            }
        }
        return true;
    }
        
     };