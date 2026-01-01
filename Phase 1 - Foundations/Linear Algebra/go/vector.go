package vector

import (
	"errors"
	"math"
)

// Vector represents a mathematical vector.
// Go uses slices for dynamic sizing (no generics for fixed-size arrays).

type Vector struct {
	components []float64
}

// New creates a new vector from components
func New(components ...float64) (*Vector, error) {
	if len(components) == 0 {
		return nil, errors.New("vector must have at least one component")
	}
	// Copy to prevent external mutation
	c := make([]float64, len(components))
	copy(c, components)
	return &Vector{components: c}, nil
}

// Dimension returns the vector's dimension
func (v *Vector) Dimension() int{
	return len(v.components)
}

// Components returns a copy of the vector's components.
func (v *Vector) Components() []float64 {
    c := make([]float64, len(v.components))
    copy(c, v.components)
    return c
}

// Add returns the sum of two vectors.
//
// Returns error if dimensions don't match.
func (v *Vector) Add(other *Vector) (*Vector, error) {
    // TODO: Check if dimensions match
    // TODO: Create result slice
    // TODO: Add corresponding components
    // TODO: Return new Vector
   if (v.Dimension() != other.Dimension()) {
   	return nil, errors.New("Dimensions don't match.")
   }
   result := make([]float64, len(v.components))
   for i, val := range v.components {
   	result[i] = val + other.components[i]
   }
   return &Vector{components: result}, nil
}