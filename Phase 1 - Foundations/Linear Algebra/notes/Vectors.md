
*Source: Linear Algebra - Ch 1 Gilbert Strang & Mathematics for Machine Learning*

## The Geometry of Linear Equations

### Linear Equations Terminology

In a linear equation with n unknowns:
- **Goal**: Determine the solution vector x
- x = column vector (unknowns)
- b = column vector (right-hand side)
- Algebraic form: Ax = b

### Row Picture vs Column Picture

**Row Picture**
- Plot each equation in n-dimensional space
- Each equation defines a line (2D), plane (3D), or hyperplane (higher dimensions)
- Solution is the intersection point of all equations

**Column Picture**
- Express system as linear combination of column vectors
- Goal: Find the right linear combination
- In the form: $x_1 \cdot \text{col}_1 + x_2 \cdot \text{col}_2 + ... = b$
- Find scalars $(x_1, x_2, ..., x_n)$ that combine columns to produce b

### Example System

```
2x - y = 0         [2  -1  0] [x]   [0]
-x + 2y + z = 1    [-1  2  1] [y] = [1]
     3y + 4z = 4   [0   3  4] [z]   [4]
                      A        x     b
```

**Each equation is a linear combination of unknowns.**

Can be rewritten as column picture:

```
   [2]     [-1]    [0]   [0]
x· [-1] + y·[2] + z·[1] = [1]
   [0]     [3]    [4]   [4]
```

**Linear Combination Solution:**
x = 0, y = 1, z = 1

### When Does a Solution Exist?

For matrix A with column vectors, when can they span the space?
- If all column vectors lie in the same plane (are coplanar), they cannot span 3D space
- For Ax = b to have a solution for every b, columns must span the full space
- This relates to whether the matrix is invertible

---

## Mathematics for Machine Learning (Ch 2)

### Linear Algebra - the study of how vectors interact with each other in vector space

In general, **vectors are objects that can be added together or multiplied by scalars** to produce another object of the same kind.

**Four ways to think about vectors:**

1) **Geometric Vector** - Directed line segments in R² or R³ with magnitude and direction

2) **Algebraic Vector** - Objects that can be added together and multiplied by scalars, following certain rules (closure, associativity, etc.)

3) **Data Representation** - A series of numbers representing features or measurements, existing in a conceptual data space

4) **Elements of Rⁿ** - Ordered n-tuples of real numbers
   - Example: $\vec{v} = \begin{bmatrix} -3 \\ 6 \end{bmatrix} \in \mathbb{R}^2$ is a vector containing 2 real numbers

**Vector Operations:**
- Adding two vectors results in another vector: A + B
- Multiplying a vector by a scalar λ results in a scaled vector: λA

---

## The Idea of Closure

**Closure** means that performing an operation on elements of a set produces another element in the same set.

For vector spaces, closure under two operations is required:

```
Vector Space Requirements
├── Closure under Vector Addition
│   └── v + w ∈ V for all v, w ∈ V
└── Closure under Scalar Multiplication
    └── λv ∈ V for all v ∈ V, λ ∈ R
```

### Geometric Interpretation

**Visual representation of vector addition:** By graphing, we can see that adding two vectors follows the parallelogram rule - place vectors tail-to-head and the sum is the diagonal. Alternatively, the order doesn't matter (commutativity): A + B = B + A.

### How are Scalars Different from Vectors?

**A Scalar is a single number.** It has magnitude only, no direction.

**A Vector has both direction and magnitude.** It can be represented as an ordered list of components.

**Coordinates and Basis:**
- Vector components like [x, y] represent how much of each basis vector is used
- The standard basis vectors in R² are î = [1, 0] and ĵ = [0, 1]
- Any vector in R² is a linear combination of î and ĵ

**Example:** [3, 2] = 3î + 2ĵ = 3[1,0] + 2[0,1]

---

## Span

**The span of a set of vectors is all possible vectors you can reach with linear combinations of those vectors.**

**Definition:** span({v₁, v₂, ..., vₖ}) = {c₁v₁ + c₂v₂ + ... + cₖvₖ | cᵢ ∈ R}

**Examples:**
- Two non-parallel 2D vectors span the entire 2D plane (R²)
- Two non-parallel 3D vectors span a plane through the origin in 3D space
- Three non-coplanar 3D vectors span the entire 3D space (R³)

**Key insight:** The span depends on whether vectors are linearly independent (not expressible as combinations of each other).
