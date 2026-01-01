"""
Robot Navigation Simulator
Apply vector concepts to robot movement.
"""
from src.vector import Vector
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import List

class Robot:
    """
    A 2D robot that moves via vector commands
    
    Attributes:
        start
        position: Current position as Vector
        path: List of all positions visited
    """
    
    def __init__(self, start_position: Vector):
        """
        Initialize robot at starting position
        
        Args:
            start_position: Initial position vector
        """
        self.position = start_position
        self.path = [self.position]
    
    def move(self, command: Vector) -> None:
        """
        Move robot by adding command vector to position.

        Args:
            command: Movement vector [dx, dy]

        Updates position and records in path.
        """
        self.position = self.position + command
        self.path.append(self.position)
    
    def apply_force(self, force: Vector) -> None:
        """
        Apply external force (like wind) to robot.

        Same as move, but semantically different.
        """
        self.position = self.position + force
        self.path.append(self.position)
    
    def distance_to(self, target: Vector) -> float:
        """
        Calculate distance to target.

        Args:
            target: Target position vector

        Returns:
            Euclidean distance to target
        """
        return (target - self.position).magnitude()
    
    def direction_to(self, target: Vector) -> Vector:
        """
        Get unit vector pointing toward target.

        Args:
            target: Target position

        Returns:
            Normalized direction vector
        """
        return (target - self.position).normalize()

class Environment:
    """
    Environment with forces that affect the robot.
    
    Attributes:
        wind: Constant wind force vector
        obstacles: List of obstacle positions
        obstacle_radius: How close robot can get to obstacle
        
    """
    def __init__(self, wind: Vector = None):
        """
        Initialize environment.

        Args:
            wind: Constant wind force (applied each step)
        """
        self.wind = wind
        self.obstacles = []
    
    def add_obstacle(self, position: Vector, radius: float = 0.5):
         """Add an obstacle at position with given radius."""
         self.obstacles.append((position, radius))
    
    def check_collision(self, robot_position: Vector) -> bool:
        """Check if robot collides with any obstacle."""
        for obstacle in self.obstacles:
            if (robot_position - obstacle[0]).magnitude() < obstacle[1]:
                return True
        return False

class Simulator:
    """
    Run navigation simulation.
    """
    def __init__(self, robot: Robot, environment: Environment, target: Vector):
        """
        Initialize simulator.

        Args:
            robot: Robot to control
            environment: Environment with forces/obstacles
            target: Goal position
        """
        self.robot = robot
        self.environment = environment
        self.target = target
        self.target_threshold = .1
        self.collision = False
    
    def step(self, command: Vector) -> bool:
        """
        Execute one simulation step.

        Args:
            command: Movement command for robot

        Returns:
            True if reached target, False otherwise

        Process:
        1. Apply command to robot
        2. Apply environment wind
        3. Check for collision
        4. Check if reached target
        """
        self.robot.move(command)
        if (self.environment.wind):
            self.robot.apply_force(self.environment.wind)
        if(self.environment.check_collision(self.robot.position)):
            self.collision = True
        return self.robot.distance_to(self.target) < self.target_threshold
    
    def run_path(self, commands: List[Vector], target_threshold: float = 0.1) -> dict:
        """
        Run a sequence of commands.

        Args:
            commands: List of movement vectors
            target_threshold: Distance to consider "reached"

        Returns:
            dict with:
            - reached: bool
            - steps: int
            - final_distance: float
            - path: list of positions
        """
        self.target_threshold = target_threshold
        result = {
            "reached" : False,
            "steps": 0,
            "collision": self.collision,
            "final_distance": 0,
            "path": self.robot.path}
        for command in commands:
            result["reached"] = self.step(command)
            result["steps"] += 1
            if self.collision:
                break
            if result["reached"]:
                break
        result["final_distance"] = (self.target - self.robot.position).magnitude()
        return result
        
    def visualize(self, show_target: bool = True, show_obstacles: bool = True):
        """
        Visualize the robot's path.

        Creates plot showing:
        - Robot path (line with markers)
        - Start position (green circle)
        - Current position (blue circle)
        - Target (red star)
        - Obstacles (gray circles)
        - Wind direction (arrow)
        """
        if self.robot.position.dimension != 2:
            raise ValueError("Visualization only supports 2D vectors")
        if show_target and self.target.dimension != 2:
            raise ValueError("Target vector must be 2D for visualization")
        if self.environment.wind and self.environment.wind.dimension != 2:
            raise ValueError("Wind vector must be 2D for visualization")

        path = self.robot.path
        path_x = [position.components[0] for position in path]
        path_y = [position.components[1] for position in path]

        fig, ax = plt.subplots()
        ax.plot(path_x, path_y, "-o", color="black", linewidth=1, markersize=4, label="Path", zorder=2)

        start = path[0]
        ax.scatter(start.components[0], start.components[1], color="green", s=80, label="Start", zorder=3)

        current = self.robot.position
        ax.scatter(current.components[0], current.components[1], color="blue", s=80, label="Current", zorder=4)

        if show_target:
            ax.scatter(
                self.target.components[0],
                self.target.components[1],
                marker="*",
                color="red",
                s=160,
                label="Target",
                zorder=5,
            )

        if show_obstacles and self.environment.obstacles:
            for index, (position, radius) in enumerate(self.environment.obstacles):
                if position.dimension != 2:
                    raise ValueError("Obstacle vector must be 2D for visualization")
                circle = Circle(
                    (position.components[0], position.components[1]),
                    radius,
                    color="gray",
                    alpha=0.4,
                    label="Obstacle" if index == 0 else None,
                    zorder=1,
                )
                ax.add_patch(circle)

        if self.environment.wind:
            wind = self.environment.wind
            anchor = start
            ax.quiver(
                anchor.components[0],
                anchor.components[1],
                wind.components[0],
                wind.components[1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color="orange",
                label="Wind",
                zorder=3,
            )

        points_x = list(path_x)
        points_y = list(path_y)

        if show_target:
            points_x.append(self.target.components[0])
            points_y.append(self.target.components[1])

        if show_obstacles:
            for position, radius in self.environment.obstacles:
                points_x.extend([position.components[0] - radius, position.components[0] + radius])
                points_y.extend([position.components[1] - radius, position.components[1] + radius])

        if self.environment.wind:
            wind_end = start + self.environment.wind
            points_x.append(wind_end.components[0])
            points_y.append(wind_end.components[1])

        x_min, x_max = min(points_x), max(points_x)
        y_min, y_max = min(points_y), max(points_y)
        span = max(x_max - x_min, y_max - y_min)
        padding = 0.1 * span if span > 0 else 1.0

        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title("Robot Navigation Path")
        ax.legend()
        plt.tight_layout()
        plt.show()


def demo_navigation():
    """
    Demonstrate the navigation system.
    
    Scenario: Robot must reach target while fighting wind.
    """
    # Setup
    robot = Robot(Vector([0, 0]))
    env = Environment(wind=Vector([0.1, 0]))  # Wind pushing right
    target = Vector([5, 3])

    # Add some obstacles
    env.add_obstacle(Vector([2, 1]), radius=0.5)
    env.add_obstacle(Vector([3, 2]), radius=0.5)

    sim = Simulator(robot, env, target)

    # Simple navigation: move toward target
    commands = []
    for _ in range(20):
        # Get direction to target and scale to step size
        direction = robot.direction_to(target)
        command = direction * 0.5  # Move 0.5 units toward target
        commands.append(command)

    results = sim.run_path(commands)
    print(f"Reached target: {results['reached']}")
    print(f"Final distance: {results['final_distance']:.2f}")

    sim.visualize()
    plt.savefig("figures/robot_navigation.png")
    plt.show()
    
class CollisionError(Exception):
    pass
    
if __name__ == "__main__":
    demo_navigation()
