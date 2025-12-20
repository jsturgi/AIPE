# Orthogonality and Projection

*Source: MIT 18.06 Linear Algebra & Mathematics for Machine Learning*

## Introduction to Orthogonality

### Angle Between Vectors

The angle θ between two vectors u and v can be found using:

$$\cos(\theta) = \frac{\vec{u} \cdot \vec{v}}{||\vec{u}|| \cdot ||\vec{v}||}$$

### Definition: Orthogonality

Two vectors are **orthogonal** if their dot product equals 0:

$$\vec{u} \perp \vec{v} \iff \vec{u} \cdot \vec{v} = 0$$

**Orthonormal vectors:** Orthogonal vectors that are also unit vectors (||u|| = ||v|| = 1)

**Note:** The zero vector is orthogonal to every vector in the vector space.

### Example

$$\vec{x} = \begin{pmatrix} 1 \\ 2 \end{pmatrix}, \quad \vec{y} = \begin{pmatrix} -4 \\ 2 \end{pmatrix}$$

$$\vec{x} \cdot \vec{y} = 1(-4) + 2(2) = -4 + 4 = 0$$

Therefore x ⊥ y (orthogonal but not orthonormal since ||x|| = √5 ≠ 1)

---

## Orthogonal Matrices

**Definition:** A square matrix Q ∈ ℝⁿˣⁿ is **orthogonal** if its columns are orthonormal to each other:

$$Q^T Q = I \implies Q^T = Q^{-1}$$

### Properties of Orthogonal Matrices

**1. Preserves Length:**
$$||Q\vec{x}|| = \sqrt{(Q\vec{x})^T(Q\vec{x})} = \sqrt{\vec{x}^T Q^T Q \vec{x}} = \sqrt{\vec{x}^T \vec{x}} = ||\vec{x}||$$

**2. Preserves Angles:**
$$\cos(\theta) = \frac{(Q\vec{x}) \cdot (Q\vec{y})}{||Q\vec{x}|| \cdot ||Q\vec{y}||} = \frac{\vec{x}^T Q^T Q \vec{y}}{||\vec{x}|| \cdot ||\vec{y}||} = \frac{\vec{x} \cdot \vec{y}}{||\vec{x}|| \cdot ||\vec{y}||}$$

**Conclusion:** Orthogonal matrices represent rotations and reflections—they preserve geometry.

### Examples of Orthogonal Matrices

**Rotation matrix:**
$$Q = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**Permutation matrix:**
$$P = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}$$

---

## Orthonormal Basis

**Definition:** A basis {v₁, v₂, ..., vₙ} is **orthonormal** if:
- ⟨vᵢ, vⱼ⟩ = 0 for i ≠ j (orthogonal)
- ||vᵢ|| = 1 for all i (normalized)

**Standard orthonormal basis in ℝ³:**
$$e_1 = \begin{pmatrix} 1 \\ 0 \\ 0 \end{pmatrix}, \quad e_2 = \begin{pmatrix} 0 \\ 1 \\ 0 \end{pmatrix}, \quad e_3 = \begin{pmatrix} 0 \\ 0 \\ 1 \end{pmatrix}$$

---

## Orthogonal Complement

**Definition:** The **orthogonal complement** of a subspace U, denoted U⊥, is the set of all vectors orthogonal to every vector in U:

$$U^{\perp} = \{\vec{w} \in V : \vec{w} \perp \vec{u} \text{ for all } \vec{u} \in U\}$$

**Properties:**
- U ∩ U⊥ = {0}
- dim(U) + dim(U⊥) = dim(V)
- (U⊥)⊥ = U

---

## Four Fundamental Subspaces (Orthogonality)

For matrix A ∈ ℝᵐˣⁿ:

| Subspace | Orthogonal Complement |
|----------|----------------------|
| Row space C(Aᵀ) ⊆ ℝⁿ | Null space N(A) ⊆ ℝⁿ |
| Column space C(A) ⊆ ℝᵐ | Left null space N(Aᵀ) ⊆ ℝᵐ |

**Key insight:** Row space ⊥ Null space

If Ax = 0, then each row of A is orthogonal to x:
$$A\vec{x} = \begin{bmatrix} \text{row}_1 \\ \vdots \\ \text{row}_m \end{bmatrix} \vec{x} = \begin{bmatrix} 0 \\ \vdots \\ 0 \end{bmatrix} \implies \text{row}_i \cdot \vec{x} = 0$$

---

## Projection

### Projection onto a Line

To project vector b onto the line through u:

$$\text{proj}_{\vec{u}}(\vec{b}) = \frac{\vec{u} \cdot \vec{b}}{\vec{u} \cdot \vec{u}} \vec{u} = \frac{\vec{u}\vec{u}^T}{\vec{u}^T\vec{u}} \vec{b}$$

**Projection matrix (onto line through u):**
$$P = \frac{\vec{u}\vec{u}^T}{\vec{u}^T\vec{u}}$$

### Example

Project onto the line through u = [1, 2, 2]ᵀ:

$$P = \frac{\vec{u}\vec{u}^T}{\vec{u}^T\vec{u}} = \frac{1}{9} \begin{pmatrix} 1 \\ 2 \\ 2 \end{pmatrix} \begin{pmatrix} 1 & 2 & 2 \end{pmatrix} = \frac{1}{9} \begin{pmatrix} 1 & 2 & 2 \\ 2 & 4 & 4 \\ 2 & 4 & 4 \end{pmatrix}$$

### Properties of Projection Matrices

1. **Idempotent:** P² = P (projecting twice = projecting once)
2. **Symmetric:** Pᵀ = P
3. **Eigenvalues:** Only 0 or 1

---

## Projection onto a Subspace

### The Problem

Given Ax = b where b is not in the column space of A, find the closest point p in C(A) to b.

**Key insight:** The error e = b - p must be orthogonal to the column space:

$$\vec{e} = \vec{b} - A\hat{x} \perp C(A)$$

This means Aᵀe = 0, which gives us:

$$A^T(\vec{b} - A\hat{x}) = 0$$
$$A^T A \hat{x} = A^T \vec{b}$$

This is the **Normal Equation**.

### Projection Matrix (onto column space of A)

$$P = A(A^T A)^{-1} A^T$$

**Projected vector:**
$$\vec{p} = P\vec{b} = A(A^T A)^{-1} A^T \vec{b}$$

**Note:** AᵀA is invertible if and only if A has independent columns.

---

## Least Squares

### Problem Setup

Fit a line y = C + Dx to points (0,1), (1,1), (2,3):

$$\begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix} \begin{bmatrix} C \\ D \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \\ 3 \end{bmatrix}$$

This system is overdetermined (3 equations, 2 unknowns). We find the least squares solution.

### Solution via Normal Equations

$$A^T A = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 1 & 1 \\ 1 & 2 \end{bmatrix} = \begin{bmatrix} 3 & 3 \\ 3 & 5 \end{bmatrix}$$

$$A^T \vec{b} = \begin{bmatrix} 1 & 1 & 1 \\ 0 & 1 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \\ 3 \end{bmatrix} = \begin{bmatrix} 5 \\ 7 \end{bmatrix}$$

Solve AᵀAx̂ = Aᵀb:
$$\begin{bmatrix} 3 & 3 \\ 3 & 5 \end{bmatrix} \begin{bmatrix} C \\ D \end{bmatrix} = \begin{bmatrix} 5 \\ 7 \end{bmatrix}$$

**Solution:** C = 1/3, D = 1 → Best fit line: y = 1/3 + x

---

## Gram-Schmidt Process

**Goal:** Convert a set of linearly independent vectors {a, b, c, ...} into an orthonormal set {q₁, q₂, q₃, ...}

### Algorithm

**Step 1:** Normalize first vector
$$\vec{q}_1 = \frac{\vec{a}}{||\vec{a}||}$$

**Step 2:** Subtract projection and normalize
$$\vec{b}' = \vec{b} - (\vec{q}_1 \cdot \vec{b})\vec{q}_1$$
$$\vec{q}_2 = \frac{\vec{b}'}{||\vec{b}'||}$$

**Step 3:** Repeat for remaining vectors
$$\vec{c}' = \vec{c} - (\vec{q}_1 \cdot \vec{c})\vec{q}_1 - (\vec{q}_2 \cdot \vec{c})\vec{q}_2$$
$$\vec{q}_3 = \frac{\vec{c}'}{||\vec{c}'||}$$

### QR Decomposition

The Gram-Schmidt process gives us the **QR decomposition**:

$$A = QR$$

where:
- Q has orthonormal columns (the q vectors)
- R is upper triangular (stores the projection coefficients)

**Why QR is useful:**
- Solving Ax = b becomes Rx = Qᵀb (easier since R is triangular)
- More numerically stable than normal equations
- Foundation for many algorithms (eigenvalue computation, least squares)

---

## Summary

| Concept | Formula |
|---------|---------|
| Orthogonality | u · v = 0 |
| Orthogonal matrix | QᵀQ = I, Q⁻¹ = Qᵀ |
| Projection onto line | P = uuᵀ/(uᵀu) |
| Projection onto C(A) | P = A(AᵀA)⁻¹Aᵀ |
| Normal equation | AᵀAx̂ = Aᵀb |
| QR decomposition | A = QR |
