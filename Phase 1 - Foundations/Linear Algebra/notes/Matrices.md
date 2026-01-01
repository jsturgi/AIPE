# Matrices

*Source: MIT Linear Algebra - Elimination of Matrices (Gilbert Strang) & Mathematics for Machine Learning*

## Elimination - Success and Failure

**Key Concepts:**
- Back Substitution
- Elimination Matrices
- Matrix Multiplication

### Basic System Example

```
x + 2y + z = 2        [1 2 1] [x]   [2]
3x + 8y + z = 12  →   [3 8 1] [y] = [12]
    4y + z = 2        [0 4 1] [z]   [2]
```

**Purpose:** Eliminate x₁, then x₂ to get upper triangular form

Ax = b

**Goal:** Transform A to U (upper triangular) where pivots cannot be zero

### To Find Determinant: Multiply the pivots

**When does elimination fail?**
- If a pivot is 0 and no row swap is possible
- If the matrix is singular (not invertible)

**If there is a non-zero below a zero pivot:** Row exchange is needed

### Augmented Matrix Elimination

```
[1  2  1 | 2]     [1  2  1 | 2]     [1  2  1 | 2]
[3  8  1 | 12] →  [0  2 -2 | 6] →   [0  2 -2 | 6]
[0  4  1 | 2]     [0  4  1 | 2]     [0  0  5 |-10]
                                        U
```

**Ux = c (Back Substitution):**
```
x + 2y + z = 2
    2y - 2z = 6       Solve for z: 5z = -10  →  z = -2
        5z = -10      Solve for y: 2y - 2(-2) = 6  →  y = 1
                      Solve for x: x + 2(1) + (-2) = 2  →  x = 2
```

**Solution:** (x, y, z) = (2, 1, -2)

### Back Substitution

Solving equations in reverse order via substitution when the system is in triangular form.

---

## Matrix Multiplication

### Column Picture

```
[1  2] [3]     [1]     [2]     [3 + 8]   [11]
[3  8] [4]  = 3[3] + 4 [8]  =  [9 + 32] = [41]
```

Matrix times column = linear combination of columns

### Row Picture

Row of A times B = Row of C

### Element-wise Definition

For matrices A (m×n), B (n×p), the product C = AB has elements:

$$C_{ij} = \sum_{k=1}^{n} A_{ik} \cdot B_{kj}$$

(Row i of A) · (Column j of B) = Element (i,j) of C

**Key Property:** Can only multiply if "neighboring" dimensions match:
```
A      B   =  C
m×k  × k×n =  m×n
     ↑ must match
```

---

## Elimination Matrices

### Example: Subtract 3 × Row 1 from Row 2

```
[1  0  0] [1  2  1]   [1  2  1]
[-3 1  0] [3  8  1] = [0  2 -2]
[0  0  1] [0  4  1]   [0  4  1]
   E₂₁        A           EA
```

### Identity Matrix

```
[1  0  0]
[0  1  0]  = I₃ (3×3 identity)
[0  0  1]
```

**Property:** AI = IA = A

### Permutation Matrices

**Exchange Rows:** Exchange the corresponding rows of the identity matrix

```
[0  1] [a  b]   [c  d]
[1  0] [c  d] = [a  b]
  P      A        PA
```

**Row operations:** Multiply on the LEFT (PA)
**Column operations:** Multiply on the RIGHT (AP)

---

## Properties of Matrix Operations

### NOT Commutative

**AB ≠ BA** in general (even when both products are defined)

```
AB = [1  2][0  0] = [2  2]
     [3  4][1  1]   [4  4]

BA = [0  0][1  2] = [0  0]
     [1  1][3  4]   [4  6]
```

### Associative Law

**(AB)C = A(BC)**

The order of operations doesn't matter, only the order of matrices.

### Distributive Law

**A(B + C) = AB + AC**
**(B + C)A = BA + CA**

### Scalar Multiplication

**λ(AB) = (λA)B = A(λB)**

---

## Transpose

**Definition:** For A ∈ ℝᵐˣⁿ, the transpose Aᵀ ∈ ℝⁿˣᵐ has elements (Aᵀ)ᵢⱼ = Aⱼᵢ

**Properties:**
- (Aᵀ)ᵀ = A
- (AB)ᵀ = BᵀAᵀ (note the reversal!)
- (A + B)ᵀ = Aᵀ + Bᵀ

### Symmetric Matrix

A matrix A is **symmetric** if Aᵀ = A

- Only square matrices can be symmetric
- Sum of symmetric matrices is symmetric
- Product of symmetric matrices is NOT necessarily symmetric

---

## Inverse Matrices

### Definition

For a square matrix A, if A⁻¹ exists:

**A⁻¹A = I** (left inverse)
**AA⁻¹ = I** (right inverse)

### Finding the Inverse via Elimination

The elimination matrix E that transforms A to I is A⁻¹:

```
E₂₁ = [1   0  0]    E₂₁⁻¹ = [1  0  0]
      [-3  1  0]            [3  1  0]
      [0   0  1]            [0  0  1]
```

To undo "subtract 3 × row 1 from row 2", we "add 3 × row 1 to row 2"

### Inverse of a Product

**(AB)⁻¹ = B⁻¹A⁻¹** (note the reversal!)

---

## Gauss-Jordan Elimination

**Method to find A⁻¹:** Augment [A | I] and reduce to [I | A⁻¹]

```
[1  3 | 1  0]     [1  3 | 1  0]     [1  0 | 7  -3]
[2  7 | 0  1]  →  [0  1 |-2  1]  →  [0  1 |-2   1]
     [A | I]                              [I | A⁻¹]
```

**Result:** A⁻¹ = [7  -3]
                  [-2   1]

**Verification:** AA⁻¹ = I

---

## Row Echelon Form (REF)

A matrix is in **row echelon form** if:

1. All zero rows are at the bottom
2. The leading entry (pivot) of each non-zero row is to the right of the pivot above it
3. All entries below a pivot are zero

### Reduced Row Echelon Form (RREF)

Additional requirements:
4. Each pivot equals 1
5. Each pivot is the only non-zero entry in its column

```
REF:                    RREF:
[1  2  3  4]           [1  0  0  *]
[0  1  2  3]    →      [0  1  0  *]
[0  0  1  2]           [0  0  1  *]
[0  0  0  0]           [0  0  0  0]
```

---

## General Solution of Ax = b

For an m×n system with m < n (more unknowns than equations):

1. **Find a particular solution** xₚ where Axₚ = b
2. **Find all solutions to Ax = 0** (the null space)
3. **General solution:** x = xₚ + xₙ where xₙ is any null space vector

### Example

```
[1  0  8  -4] [x₁]   [42]
[0  1  2  12] [x₂] = [8]
              [x₃]
              [x₄]
```

**Particular solution:** Set free variables x₃ = x₄ = 0, solve for x₁, x₂
**Null space:** Find vectors where Ax = 0

**General solution:**
```
x = [42]     [-8]      [4]
    [8]  + λ [-2]  + μ [-12]
    [0]      [1]       [0]
    [0]      [0]       [1]
```

---

## Matrix Rank and Null Space

**Rank(A)** = number of pivots = dimension of column space

**Null Space** = {x : Ax = 0}

**Nullity(A)** = dimension of null space = n - rank(A)

**Rank-Nullity Theorem:** rank(A) + nullity(A) = n (number of columns)

---

## Block Matrices

Matrices can be partitioned into blocks:

```
[A₁  A₂] [B₁  B₂]   [A₁B₁+A₂B₃  A₁B₂+A₂B₄]
[A₃  A₄] [B₃  B₄] = [A₃B₁+A₄B₃  A₃B₂+A₄B₄]
```

Block multiplication follows the same rules as element multiplication, as long as the inner dimensions match.

---

## Groups (MML 2.4-2.5)

**Definition:** A set G with operation ⊗ forms a group (G, ⊗) if:

1. **Closure:** ∀x, y ∈ G: x ⊗ y ∈ G
2. **Associativity:** ∀x, y, z ∈ G: (x ⊗ y) ⊗ z = x ⊗ (y ⊗ z)
3. **Neutral element:** ∃e ∈ G: e ⊗ x = x ⊗ e = x for all x
4. **Inverse element:** ∀x ∈ G, ∃y ∈ G: x ⊗ y = y ⊗ x = e

**Abelian Group:** If additionally x ⊗ y = y ⊗ x (commutative), the group is Abelian.

**Example:** (ℝⁿˣⁿ invertible matrices, ×) forms a group (General Linear Group GL(n))
- Not Abelian since matrix multiplication is not commutative
