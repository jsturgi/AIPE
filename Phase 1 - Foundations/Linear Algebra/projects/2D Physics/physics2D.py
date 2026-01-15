"""
Simple 2D Physics Simulator
Vectors as physical quantities
"""
from src.vector import Vector
import matplotlib.pyplot as plt
from typing import List

class Particle:
    """
    A particle with position, velocity, and acceleration.

    Physics update rule:
        velocity += acceleration * dt
        position += velocity * dt
    """

    def __init__(self, position: Vector, velocity: Vector, mass: float = 1.0):
        """
        Initialize particle.

        Args:
            position: Initial position [x, y]
            velocity: Initial velocity [vx, vy]
            mass: Mass of particle (for force calculations)
        """
        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.acceleration = Vector([0,0])
        self.trajectory = []
    
    def apply_force(self, force: Vector) -> None:
        """
        Apply a force to the particle.

        F = ma, so a = F/m
        Forces accumulate until cleared.
        """
        self.acceleration = force / self.mass
    
    def clear_forces(self) -> None:
        """Clear accumulated forces (call after update)."""
        self.acceleration = Vector([0,0])
    
    def update(self, dt: float) -> None:
        """
        Update particle for one timestep.

        Args:
            dt: Time step size

        Physics:
            v_new = v + a * dt
            p_new = p + v_new * dt
        """
        # TODO: Update velocity
        # TODO: Update position
        # TODO: Record position in trajectory
        # TODO: Clear forces
        self.velocity = self.velocity + self.acceleration * dt
        self.position = self.position + self.velocity * dt
        self.trajectory.append(self.position)
        self.clear_forces()
    
    def kinetic_energy(self) -> float:
        """Calculate kinetic energy: KE = 0.5 * m * v^2"""
        
        return .5 * self.mass * self.velocity.dot(self.velocity)

class PhysicsWorld:
    """
    A 2D physics world with gravity and particles.
    """

    def __init__(self, gravity: Vector = Vector([0, -9.8])):
        """
        Initialize world.

        Args:
            gravity: Gravity vector (default: [0, -9.8])
        """
        self.gravity = gravity
        self.particles = []
    
    def add_particle(self, particle: Particle) -> None:
        """Add a particle to the world."""
        self.particles.append(particle)
    
    def step(self, dt: float = 0.01) -> None:
        """
        Advance simulation by one timestep.

        For each particle:
        1. Apply gravity force
        2. Apply any other forces
        3. Update particle
        """
        # TODO: For each particle: apply gravity, update
        for particle in self.particles:
            particle.apply_force(self.gravity)
            particle.update(dt)
    
    def run(self, duration: float, dt: float = 0.01) -> None:
        """
        Run simulation for given duration.

        Args:
            duration: Total time to simulate
            dt: Timestep size
        """
        t = 0
        while t < duration:
            self.step(dt)
            t += dt
    
    def visualize(self):
        """
        Visualize all particle trajectories.
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        for i, particle in enumerate(self.particles):
            trajectory = particle.trajectory
            xs = [p.components[0] for p in trajectory]
            ys = [p.components[1] for p in trajectory]
            ax.plot(xs, ys, '-', label=f'Particle {i+1}')
            ax.scatter(xs[0], ys[0], marker='o', s=100)  # Start
            ax.scatter(xs[-1], ys[-1], marker='x', s=100)  # End

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title('Projectile Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        return fig

def demo_projectile_motion():
    """
    Demonstrate projectile motion.

    Launch projectiles at different angles, same speed.
    """
    world = PhysicsWorld(gravity=Vector([0, -9.8]))

    # Launch speed
    speed = 20

    # Different launch angles
    angles = [30, 45, 60, 75]

    for angle in angles:
        import math
        rad = math.radians(angle)
        vx = speed * math.cos(rad)
        vy = speed * math.sin(rad)

        particle = Particle(
            position=Vector([0, 0]),
            velocity=Vector([vx, vy])
        )
        world.add_particle(particle)

    # Run until all particles hit ground
    world.run(duration=4, dt=0.01)

    fig = world.visualize()
    fig.savefig("figures/projectile_motion.png")
    plt.show()

if __name__ == "__main__":
    demo_projectile_motion()
        
            
    