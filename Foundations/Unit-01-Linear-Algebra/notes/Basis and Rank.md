# Basis and Rank

*Source: Mathematics for Machine Learning & MIT 18.06*

## Definitions and Fundamental Concepts

### Spanning Set

**Definition:** Consider a vector space V and a set of vectors {x₁, x₂, ..., xₖ} ∈ V. If every vector v ∈ V can be expressed as a linear combination of the xᵢ's, then {x₁, ..., xₖ} is called a **spanning set** of V.

- The set of all linear combinations is called the **span** of the vectors
- We write V = span{x₁, ..., xₖ} or V = span(A) where A is a matrix with xᵢ as columns

### Basis

**Definition:** A **basis** of a vector space V is a set of vectors B = {b₁, b₂, ..., bₙ} that satisfies:
1. B is linearly independent
2. B spans V

Equivalently: **A basis is a minimal spanning set and a maximal linearly independent set of vectors.**

The following statements are equivalent:
- B is a basis of V
- B is a minimal spanning set (removing any vector makes it no longer span V)
- B is a maximal linearly independent set (adding any vector makes it linearly dependent)

**Key Property:** Every vector v ∈ V can be written as a **unique** linear combination of basis vectors:
$$v = \lambda_1 b_1 + \lambda_2 b_2 + ... + \lambda_n b_n$$

where the λᵢ are unique scalars called the **coordinates** of v with respect to basis B.

### Standard Basis in ℝⁿ

In ℝ³, the canonical/standard basis is:

$$\mathcal{B} = \left\{ \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix} \right\} = \{e_1, e_2, e_3\}$$

**Note:** A vector space can have many different bases, but every basis has the same number of vectors.

### Dimension

**Definition:** The **dimension** of a vector space V is the number of vectors in any basis of V.

$$\dim(V) = |B| \text{ for any basis } B \text{ of } V$$

- dim(ℝⁿ) = n
- dim({0}) = 0 (the trivial space)

---

## Finding a Basis

To find a basis for a subspace U = span{x₁, ..., xₖ} ⊆ ℝⁿ:

1. Write the spanning vectors as **columns** of a matrix A
2. Reduce A to **row echelon form**
3. The **pivot columns** of the original matrix form a basis of U

### Example

Find a basis for span{x₁, x₂, x₃} where:

$$A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 7 \\ 3 & 6 & 10 \end{bmatrix} \xrightarrow{REF} \begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}$$

Pivots are in columns 1 and 3, so a basis is {x₁, x₃}:

$$\text{Basis} = \left\{ \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \begin{bmatrix} 3 \\ 7 \\ 10 \end{bmatrix} \right\}$$

---

## Rank

**Definition:** The **rank** of a matrix A ∈ ℝᵐˣⁿ equals the number of linearly independent columns of A (equivalently, the number of pivot columns in REF).

$$\text{rank}(A) = \dim(\text{column space of } A)$$

### Properties of Rank

1. **Upper bound:** rank(A) ≤ min{m, n}
   - Can't have more independent columns than rows, or vice versa

2. **Column rank = Row rank:** rank(A) = rank(Aᵀ)
   - Number of independent columns = Number of independent rows

3. **Rank-Nullity Theorem:** For A ∈ ℝᵐˣⁿ:
   $$\text{rank}(A) + \text{nullity}(A) = n$$
   where nullity(A) = dim(null space of A) = number of free variables

### Full Rank

A matrix A ∈ ℝᵐˣⁿ has **full rank** if:
$$\text{rank}(A) = \min\{m, n\}$$

**Consequences of full rank:**
- If m ≤ n (more columns than rows): rank = m, at least one solution exists for every b
- If m ≥ n (more rows than columns): rank = n, at most one solution exists
- If m = n and full rank: A is invertible, exactly one solution exists

**If not full rank:**
- Null space is non-trivial (contains more than just 0)
- There will be free variables in the solution
- Either no solution or infinitely many solutions to Ax = b

---

## Four Fundamental Subspaces

For a matrix A ∈ ℝᵐˣⁿ:

| Subspace | Definition | Dimension |
|----------|------------|-----------|
| **Column Space** C(A) | span of columns of A | rank(A) = r |
| **Row Space** C(Aᵀ) | span of rows of A | rank(A) = r |
| **Null Space** N(A) | {x : Ax = 0} | n - r |
| **Left Null Space** N(Aᵀ) | {y : Aᵀy = 0} | m - r |

### Relationships

- Column space ⊆ ℝᵐ, Null space ⊆ ℝⁿ
- Row space ⊥ Null space (orthogonal complements in ℝⁿ)
- Column space ⊥ Left null space (orthogonal complements in ℝᵐ)

---

## Summary

| Concept | Definition |
|---------|------------|
| Spanning Set | Vectors whose linear combinations cover a space |
| Basis | Minimal spanning set = Maximal independent set |
| Dimension | Number of vectors in a basis |
| Rank | Number of independent columns = Number of pivots |
| Full Rank | rank(A) = min{m, n} |
| Nullity | Dimension of null space = n - rank(A) |
