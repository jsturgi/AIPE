# Vector Spaces

*Source: Mathematics for Machine Learning 2.4-2.5*

## Groups

Consider a set G and an operation ⊗: G × G → G. Then G = (G, ⊗) is called a **group** if the following hold:

1. **Closure under ⊗:** ∀x, y ∈ G: x ⊗ y ∈ G

2. **Associativity:** ∀x, y, z ∈ G: (x ⊗ y) ⊗ z = x ⊗ (y ⊗ z)

3. **Neutral element:** ∃e ∈ G: ∀x ∈ G: e ⊗ x = x ⊗ e = x

4. **Inverse element:** ∀x ∈ G, ∃y ∈ G: x ⊗ y = y ⊗ x = e
   We write x⁻¹ to denote the inverse element of x

5. **Abelian Group:** If additionally ∀x, y ∈ G: x ⊗ y = y ⊗ x, then G = (G, ⊗) is an **Abelian group** (commutative)

---

## Common Number Sets and Groups

| Set | Symbol | Description |
|-----|--------|-------------|
| Natural numbers | ℕ | {1, 2, 3, ...} |
| Integers | ℤ | {..., -2, -1, 0, 1, 2, ...} |
| Rational numbers | ℚ | Fractions p/q where p, q ∈ ℤ |
| Real numbers | ℝ | All points on the number line |
| Complex numbers | ℂ | a + bi where i² = -1 |

**Examples:**
- **(ℤ, +)** is an Abelian group
- **(ℕ, +)** is NOT a group (inverse elements are missing)
- **(ℤ, ×)** is NOT a group (inverse elements missing for most integers)
- **(ℝ, ×)** is NOT a group (0 doesn't have an inverse element)
- **(ℝ\{0}, ×)** IS a group (removing 0 fixes the inverse problem)
- **(ℝⁿˣⁿ, +)** the set of n×n matrices is an Abelian group under addition
- **(ℝⁿˣⁿ, ×)** the set of n×n matrices:
  - Closure: ✓ (matrix multiplication of n×n matrices gives n×n matrix)
  - Neutral element: ✓ (identity matrix Iₙ)
  - Inverse element: ✗ (not every matrix has an inverse)

---

## General Linear Group

**Definition:** The set of regular (invertible) matrices A ∈ ℝⁿˣⁿ forms a group with respect to matrix multiplication called the **General Linear Group GL(n, ℝ)**.

Since matrix multiplication is not commutative, this group is NOT Abelian.

---

## Vector Space Definition

**Definition:** A real-valued **vector space** V = (V, +, ·) is a set V with two operations:

- **+: V × V → V** (inner operation: vector addition)
- **·: ℝ × V → V** (outer operation: scalar multiplication)

where:

### 1. (V, +) is an Abelian Group
- Closure: x + y ∈ V
- Associativity: (x + y) + z = x + (y + z)
- Neutral element: ∃0 ∈ V: x + 0 = x
- Inverse element: ∀x ∃(-x): x + (-x) = 0
- Commutativity: x + y = y + x

### 2. Distributivity
- ∀λ ∈ ℝ, x, y ∈ V: λ·(x + y) = λ·x + λ·y
- ∀λ, ψ ∈ ℝ, x ∈ V: (λ + ψ)·x = λ·x + ψ·x

### 3. Associativity (outer operation)
- ∀λ, ψ ∈ ℝ, x ∈ V: λ·(ψ·x) = (λψ)·x

### 4. Neutral Element (outer operation)
- ∀x ∈ V: 1·x = x

---

## ℝⁿ - The Standard Vector Space

**ℝⁿ** is the set of n-tuples of real numbers:

$$\mathbb{R}^n = \left\{ \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} : x_i \in \mathbb{R} \right\}$$

- x denotes a column vector
- xᵢ denotes the i-th component of x

---

## Vector Subspace

**Definition:** Let V = (V, +, ·) be a vector space and U ⊆ V, U ≠ ∅. Then U = (U, +, ·) is called a **vector subspace** (or linear subspace) of V if U is a vector space with the vector space operations + and · inherited from V.

### Conditions for Subspace

To determine whether U is a subspace of V, we need to show:

1. **U ≠ ∅**, in particular: **0 ∈ U** (the zero vector must be in U)

2. **Closure of U:**
   - Under scalar multiplication: ∀λ ∈ ℝ, x ∈ U: λ·x ∈ U
   - Under vector addition: ∀x, y ∈ U: x + y ∈ U

**Note:** Every vector space V has at least two subspaces: V itself and {0} (the trivial subspace).

---

## Solution Sets as Subspaces

The solution set of a **homogeneous** system of linear equations Ax = 0 forms a subspace (called the **null space** or **kernel** of A).

Example: For
```
A = [2  3]
    [4  6]
```

The null space is the set of all x such that Ax = 0.

**Important:** The solution set of Ax = b (when b ≠ 0) is NOT a subspace because it doesn't contain the zero vector.

---

## Linear Combinations

**Definition:** A **linear combination** of vectors x₁, x₂, ..., xₖ is any vector of the form:

$$\lambda_1 x_1 + \lambda_2 x_2 + ... + \lambda_k x_k$$

where λᵢ ∈ ℝ are scalars.

### Span

**Definition:** The **span** of a set of vectors {x₁, x₂, ..., xₖ} is the set of all linear combinations:

$$\text{span}\{x_1, ..., x_k\} = \left\{ \sum_{i=1}^{k} \lambda_i x_i : \lambda_i \in \mathbb{R} \right\}$$

**Key Properties:**
- The span of any set of vectors is always a subspace
- span{x₁, ..., xₖ} is the smallest subspace containing x₁, ..., xₖ

### Example

```
span{[1], [0]}  = ℝ² (the entire 2D plane)
     [0]  [1]

span{[1], [2]}  = a line through the origin (since [2] = 2·[1])
     [1]  [2]                                      [2]     [1]
```

---

## Linear Independence

**Definition:** Vectors x₁, x₂, ..., xₖ are **linearly independent** if the only solution to:

$$\lambda_1 x_1 + \lambda_2 x_2 + ... + \lambda_k x_k = 0$$

is λ₁ = λ₂ = ... = λₖ = 0 (the trivial solution).

If a non-trivial solution exists (some λᵢ ≠ 0), the vectors are **linearly dependent**.

### Geometric Interpretation
- 2 vectors are linearly dependent ⟺ they are parallel (one is a scalar multiple of the other)
- 3 vectors in ℝ³ are linearly dependent ⟺ they are coplanar

---

## Summary

| Concept | Definition |
|---------|------------|
| Vector Space | Set with vector addition and scalar multiplication satisfying 8 axioms |
| Subspace | Subset that is itself a vector space (closed under + and ·, contains 0) |
| Linear Combination | Sum of scalar multiples of vectors |
| Span | Set of all linear combinations of given vectors |
| Linear Independence | No vector can be written as combination of others |
