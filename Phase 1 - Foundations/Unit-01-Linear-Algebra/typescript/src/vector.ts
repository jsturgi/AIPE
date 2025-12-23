/**
 * Vector class for linear algebra operations.
 * Typescript provides compile-time type checking.
 */
 
export class Vector 
{
  private readonly _components: readonly number[];
   
  constructor(components: number[]) {
    if (components.length === 0) {
      throw new Error("Vector must have at least one component")
    }
    if (!components.every(c => typeof c === 'number' &&
      !isNaN(c))) {
      throw new Error("All components must be valid numbers");
    }
  }
     
  get components(): readonly number[] 
  {
    return this._components;
  }
  
  /** 
   * Returns the dimension (number of components) of the vector.
   * 
   * @returns The dimension of the Vector
   */
   get dimension(): number
   {
     return this._components.length;
   }
   /**
    * Adds another vector to this vector.
    *
    * Vectors must have the same dimension.
    * Formula: (a₁, a₂, ..., aₙ) + (b₁, b₂, ..., bₙ) = (a₁+b₁, a₂+b₂, ..., aₙ+bₙ)
    *
    * @param other - The vector to add
    * @returns A new vector representing the sum
    * @throws Error if dimensions don't match
    */
    add(other: Vector): Vector
    {
      if (other.dimension != this.dimension)
      {
        throw new Error("Dimensions don't match.");
      }
      let arr = this._components.map((val, i) => val + other._components[i]);
      return new Vector(arr);
    }
    
    subtract(other: Vector): Vector 
    {
        // TODO: Validate dimensions
        // TODO: Compute element-wise differences
        // TODO: Return new Vector
        if (other.dimension != this.dimension)
        {
          throw new Error("Dimensions don't match.");
        }
        let arr = this._components.map((val, i) => val - other._components[i]);
        return new Vector(arr);
    }
    /**
     * Multiplies the vector by a scalar.
     *
     * Formula: c * (a₁, a₂, ..., aₙ) = (c*a₁, c*a₂, ..., c*aₙ)
     *
     * @param scalar - The scalar to multiply by
     * @returns A new vector scaled by the scalar
     */
    scale(scalar: number): Vector 
    {
      let arr = this._components.map((val) => val * scalar);
      return new Vector(arr);   
    }
    
    /**
     * Computes the dot product with another vector.
     *
     * Formula: a · b = a₁*b₁ + a₂*b₂ + ... + aₙ*bₙ
     *
     * @param other - The vector to compute dot product with
     * @returns The dot product (a scalar value)
     * @throws Error if dimensions don't match
     */
     dot(other: Vector): number {
       if (other.dimension != this.dimension)
       {
         throw new Error("Dimensions don't match.");
       }
       let arr = this._components.map((val, i) => val * other._components[i]);
       return arr.reduce((acc, val) => acc + val, 0);
     }
    
     /**
      * Computes the Euclidean magnitude (length) of the vector.
      *
      * Formula: ||v|| = √(v₁² + v₂² + ... + vₙ²)
      *
      * @returns The magnitude of the vector
      */
     magnitude(): number {
         // TODO: Compute sum of squares of components
         // TODO: Return square root of sum
       return Math.sqrt(this._components.reduce((acc, c) => acc + c * c, 0));
     }
     
     /**
      * Returns a unit vector in the same direction.
      *
      * Formula: v̂ = v / ||v||
      *
      * @returns A normalized vector (magnitude = 1)
      * @throws Error if vector is zero vector
      */
     normalize(): Vector {
         // TODO: Compute magnitude
         // TODO: Check if magnitude is zero (within tolerance)
         // TODO: Divide each component by magnitude
         // TODO: Return new Vector
       let magnitude = this.magnitude();
       if (magnitude === 0)
       {
         throw Error("Can't normalize a zero vector");
       }
       return new Vector(this._components.map(val => val / magnitude));    
     }
     
     /**
      * Checks if two vectors are approximately equal.
      *
      * @param other - The vector to compare with
      * @param tolerance - The maximum difference allowed (default: 1e-10)
      * @returns True if vectors are approximately equal
      */
      equals(other: Vector, tolerance: number = 1e-10): boolean 
      {
          // TODO: Check if dimensions match
          // TODO: Check each component is within tolerance
          // TODO: Return true only if all components match
          if (other.dimension != this.dimension)
          {
            throw new Error("Dimensions don't match.");
          }
        return this._components.every((val, i) => val === other._components[i]);
      }
      
      toString(): string {
        return `Vector([${this._components.join(',')}])`;
      }
     
}