import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

class FlightPattern(Enum):
    COORDINATED_ATTACK = "coordinated_attack"
    RANDOM_DISPERSAL = "random_dispersal" 
    FORMATION_FLYING = "formation_flying"
    EVASIVE_MANEUVERS = "evasive_maneuvers"
    PERIMETER_SWEEP = "perimeter_sweep"

@dataclass
class DroneSpec:
    """Drone physical and performance specifications"""
    wingspan: float  # meters
    max_velocity: float  # m/s
    max_acceleration: float  # m/s²
    turn_radius: float  # meters
    mass: float  # kg

@dataclass
class SimulationBounds:
    """3D simulation volume boundaries"""
    x_min: float = 0.0
    x_max: float = 1000.0  # 1km
    y_min: float = 0.0
    y_max: float = 1000.0  # 1km
    z_min: float = 50.0    # Minimum altitude
    z_max: float = 500.0   # Maximum altitude

class DroneSwarmGenerator:
    """
    Generates realistic 3D trajectories for drone swarms with various flight patterns.
    Optimized for desktop simulation with configurable complexity.
    """
    
    def __init__(self, bounds: SimulationBounds):
        self.bounds = bounds
        
        # Predefined drone types
        self.drone_types = {
            'micro': DroneSpec(0.3, 15.0, 3.0, 5.0, 0.5),
            'small': DroneSpec(0.8, 25.0, 4.0, 8.0, 1.2), 
            'medium': DroneSpec(1.5, 35.0, 5.0, 12.0, 2.5)
        }
        
        # Environmental parameters
        self.wind_velocity = np.array([2.0, 1.0, 0.5])  # m/s in x,y,z
        self.noise_std = 0.5  # Position noise standard deviation
        
    def generate_swarm_trajectories(self, 
                                  num_drones: int,
                                  pattern: FlightPattern,
                                  duration: float = 60.0,
                                  timestep: float = 0.1,
                                  drone_type: str = 'small') -> dict:
        """
        Generate trajectories for entire drone swarm
        
        Args:
            num_drones: Number of drones in swarm
            pattern: Flight pattern type
            duration: Simulation duration in seconds
            timestep: Time step in seconds
            drone_type: Type of drone ('micro', 'small', 'medium')
            
        Returns:
            Dictionary containing trajectories and metadata
        """
        spec = self.drone_types[drone_type]
        time_steps = int(duration / timestep)
        times = np.linspace(0, duration, time_steps)
        
        # Initialize trajectory storage
        trajectories = np.zeros((num_drones, time_steps, 3))  # [drone, time, xyz]
        velocities = np.zeros((num_drones, time_steps, 3))
        
        # Generate initial positions
        initial_positions = self._generate_initial_positions(num_drones, pattern)
        
        # Generate trajectories based on pattern
        for drone_id in range(num_drones):
            traj, vel = self._generate_single_trajectory(
                drone_id, initial_positions[drone_id], spec, 
                pattern, times, timestep
            )
            trajectories[drone_id] = traj
            velocities[drone_id] = vel
            
        return {
            'trajectories': trajectories,
            'velocities': velocities,
            'times': times,
            'spec': spec,
            'pattern': pattern,
            'num_drones': num_drones,
            'bounds': self.bounds
        }
    
    def _generate_initial_positions(self, num_drones: int, pattern: FlightPattern) -> np.ndarray:
        """Generate starting positions based on flight pattern"""
        positions = np.zeros((num_drones, 3))
        
        if pattern == FlightPattern.COORDINATED_ATTACK:
            # Start from perimeter, converge to center
            angles = np.linspace(0, 2*np.pi, num_drones, endpoint=False)
            radius = 400.0  # Start 400m from center
            center_x, center_y = 500.0, 500.0
            
            positions[:, 0] = center_x + radius * np.cos(angles)
            positions[:, 1] = center_y + radius * np.sin(angles)
            positions[:, 2] = np.random.uniform(100, 300, num_drones)
            
        elif pattern == FlightPattern.FORMATION_FLYING:
            # V-formation or grid formation
            if num_drones <= 10:
                # V-formation
                positions = self._generate_v_formation(num_drones)
            else:
                # Grid formation
                positions = self._generate_grid_formation(num_drones)
                
        elif pattern == FlightPattern.RANDOM_DISPERSAL:
            # Random starting positions
            positions[:, 0] = np.random.uniform(self.bounds.x_min + 50, self.bounds.x_max - 50, num_drones)
            positions[:, 1] = np.random.uniform(self.bounds.y_min + 50, self.bounds.y_max - 50, num_drones)
            positions[:, 2] = np.random.uniform(self.bounds.z_min + 20, self.bounds.z_max - 20, num_drones)
            
        elif pattern == FlightPattern.EVASIVE_MANEUVERS:
            # Start clustered, then spread out
            center = np.array([200.0, 200.0, 200.0])
            cluster_radius = 50.0
            positions = center + np.random.normal(0, cluster_radius, (num_drones, 3))
            
        elif pattern == FlightPattern.PERIMETER_SWEEP:
            # Start along one edge
            positions[:, 0] = self.bounds.x_min + 20
            positions[:, 1] = np.linspace(self.bounds.y_min + 50, self.bounds.y_max - 50, num_drones)
            positions[:, 2] = np.random.uniform(150, 250, num_drones)
            
        return positions
    
    def _generate_single_trajectory(self, drone_id: int, start_pos: np.ndarray, 
                                  spec: DroneSpec, pattern: FlightPattern,
                                  times: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate trajectory for single drone"""
        num_steps = len(times)
        trajectory = np.zeros((num_steps, 3))
        velocity = np.zeros((num_steps, 3))
        
        trajectory[0] = start_pos
        current_pos = start_pos.copy()
        current_vel = np.zeros(3)
        
        for i in range(1, num_steps):
            t = times[i]
            
            # Calculate desired velocity based on pattern
            desired_vel = self._calculate_desired_velocity(
                drone_id, current_pos, current_vel, t, pattern, spec
            )
            
            # Apply physical constraints
            desired_vel = self._apply_velocity_constraints(desired_vel, current_vel, spec, dt)
            
            # Add environmental effects
            effective_vel = desired_vel + self.wind_velocity * 0.1  # Partial wind effect
            
            # Update position
            current_pos += effective_vel * dt
            current_vel = effective_vel
            
            # Apply boundary constraints
            current_pos = self._apply_boundary_constraints(current_pos)
            
            # Add sensor noise
            noisy_pos = current_pos + np.random.normal(0, self.noise_std, 3)
            
            trajectory[i] = noisy_pos
            velocity[i] = current_vel
            
        return trajectory, velocity
    
    def _calculate_desired_velocity(self, drone_id: int, pos: np.ndarray, 
                                  vel: np.ndarray, t: float, 
                                  pattern: FlightPattern, spec: DroneSpec) -> np.ndarray:
        """Calculate desired velocity based on flight pattern"""
        
        if pattern == FlightPattern.COORDINATED_ATTACK:
            # Move toward center with slight spiral
            center = np.array([500.0, 500.0, 200.0])
            to_center = center - pos
            distance = np.linalg.norm(to_center)
            
            if distance > 50:
                # Add spiral component
                spiral_vel = np.array([-to_center[1], to_center[0], 0]) * 0.1
                desired = (to_center / distance) * spec.max_velocity * 0.7 + spiral_vel
            else:
                # Hover near center
                desired = np.random.normal(0, 2.0, 3)
                
        elif pattern == FlightPattern.FORMATION_FLYING:
            # Maintain formation while moving forward
            formation_vel = np.array([spec.max_velocity * 0.6, 0, 0])
            
            # Add formation-keeping forces
            neighbor_force = self._calculate_formation_forces(drone_id, pos)
            desired = formation_vel + neighbor_force
            
        elif pattern == FlightPattern.RANDOM_DISPERSAL:
            # Random walk with momentum
            random_component = np.random.normal(0, spec.max_velocity * 0.3, 3)
            momentum_component = vel * 0.7
            desired = random_component + momentum_component
            
        elif pattern == FlightPattern.EVASIVE_MANEUVERS:
            # Zigzag pattern with random direction changes
            if np.random.random() < 0.05:  # 5% chance to change direction
                self._evasive_direction = np.random.normal(0, 1, 3)
                self._evasive_direction = self._evasive_direction / np.linalg.norm(self._evasive_direction)
            
            if not hasattr(self, '_evasive_direction'):
                self._evasive_direction = np.array([1, 0, 0])
                
            base_vel = self._evasive_direction * spec.max_velocity * 0.8
            zigzag = np.array([0, np.sin(t * 3) * 5, np.cos(t * 2) * 3])
            desired = base_vel + zigzag
            
        elif pattern == FlightPattern.PERIMETER_SWEEP:
            # Sweep across the area
            sweep_vel = np.array([spec.max_velocity * 0.5, 0, 0])
            
            # Add slight weaving
            weave = np.array([0, np.sin(t + drone_id) * 3, 0])
            desired = sweep_vel + weave
            
        else:
            desired = np.zeros(3)
            
        return desired
    
    def _apply_velocity_constraints(self, desired_vel: np.ndarray, current_vel: np.ndarray,
                                  spec: DroneSpec, dt: float) -> np.ndarray:
        """Apply physical constraints to velocity changes"""
        # Limit acceleration
        vel_change = desired_vel - current_vel
        max_vel_change = spec.max_acceleration * dt
        
        if np.linalg.norm(vel_change) > max_vel_change:
            vel_change = vel_change / np.linalg.norm(vel_change) * max_vel_change
            
        new_vel = current_vel + vel_change
        
        # Limit maximum velocity
        if np.linalg.norm(new_vel) > spec.max_velocity:
            new_vel = new_vel / np.linalg.norm(new_vel) * spec.max_velocity
            
        return new_vel
    
    def _apply_boundary_constraints(self, pos: np.ndarray) -> np.ndarray:
        """Keep drones within simulation boundaries"""
        pos[0] = np.clip(pos[0], self.bounds.x_min + 10, self.bounds.x_max - 10)
        pos[1] = np.clip(pos[1], self.bounds.y_min + 10, self.bounds.y_max - 10)
        pos[2] = np.clip(pos[2], self.bounds.z_min + 10, self.bounds.z_max - 10)
        return pos
    
    def _calculate_formation_forces(self, drone_id: int, pos: np.ndarray) -> np.ndarray:
        """Calculate forces to maintain formation (simplified)"""
        # This is a placeholder - you can extend this for complex formations
        return np.random.normal(0, 1.0, 3)
    
    def _generate_v_formation(self, num_drones: int) -> np.ndarray:
        """Generate V-formation starting positions"""
        positions = np.zeros((num_drones, 3))
        
        # Leader at front
        positions[0] = [200, 500, 200]
        
        # Followers in V shape
        for i in range(1, num_drones):
            side = 1 if i % 2 == 0 else -1  # Alternate sides
            row = (i + 1) // 2
            
            positions[i] = [
                200 - row * 20,  # Move back
                500 + side * row * 15,  # Move to side
                200 + np.random.uniform(-10, 10)  # Small altitude variation
            ]
            
        return positions
    
    def _generate_grid_formation(self, num_drones: int) -> np.ndarray:
        """Generate grid formation starting positions"""
        grid_size = int(np.ceil(np.sqrt(num_drones)))
        positions = np.zeros((num_drones, 3))
        
        spacing = 30.0  # meters between drones
        start_x, start_y = 200.0, 200.0
        
        for i in range(num_drones):
            row = i // grid_size
            col = i % grid_size
            
            positions[i] = [
                start_x + col * spacing,
                start_y + row * spacing,
                200 + np.random.uniform(-5, 5)
            ]
            
        return positions

def visualize_trajectories(trajectory_data: dict, max_drones_display: int = 10):
    """Visualize drone trajectories in 3D"""
    trajectories = trajectory_data['trajectories']
    times = trajectory_data['times']
    bounds = trajectory_data['bounds']
    
    # Larger figure size to accommodate 3D plot and labels
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Display subset of trajectories to avoid clutter
    num_display = min(max_drones_display, trajectories.shape[0])
    colors = plt.cm.tab10(np.linspace(0, 1, num_display))
    
    for i in range(num_display):
        traj = trajectories[i]
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
               color=colors[i], alpha=0.7, linewidth=1.5,
               label=f'Drone {i+1}')
        
        # Mark start and end points
        ax.scatter(*traj[0], color=colors[i], s=100, marker='o', alpha=0.8)
        ax.scatter(*traj[-1], color=colors[i], s=100, marker='s', alpha=0.8)
    
    # Set bounds and labels
    ax.set_xlim(bounds.x_min, bounds.x_max)
    ax.set_ylim(bounds.y_min, bounds.y_max)
    ax.set_zlim(bounds.z_min, bounds.z_max)
    
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('')  # Remove default Z label
    ax.set_title(f'Drone Swarm Trajectories - {trajectory_data["pattern"].value}\n'
                f'{trajectory_data["num_drones"]} drones, {len(times):.1f}s duration')
    
    # Manually place Z-axis label where it won't be cut off
    ax.text2D(0.02, 0.5, 'Z (meters)', transform=ax.transAxes, 
              rotation=90, va='center', ha='center', fontsize=12)
    
    if num_display <= 10:
        ax.legend()
    
    return fig

# Example usage and testing
if __name__ == "__main__":
    # Initialize generator
    bounds = SimulationBounds()
    generator = DroneSwarmGenerator(bounds)
    
    # Test different scenarios
    scenarios = [
        (10, FlightPattern.COORDINATED_ATTACK, "Coordinated Attack"),
        (15, FlightPattern.FORMATION_FLYING, "Formation Flying"),
        (20, FlightPattern.RANDOM_DISPERSAL, "Random Dispersal"),
        (12, FlightPattern.EVASIVE_MANEUVERS, "Evasive Maneuvers")
    ]
    
    print("Generating drone trajectories...")
    
    for num_drones, pattern, description in scenarios:
        print(f"\nScenario: {description}")
        print(f"Drones: {num_drones}, Pattern: {pattern.value}")
        
        # Generate trajectories
        data = generator.generate_swarm_trajectories(
            num_drones=num_drones,
            pattern=pattern,
            duration=30.0,  # 30 second simulation
            timestep=0.1,
            drone_type='small'
        )
        
        # Basic statistics
        trajectories = data['trajectories']
        velocities = data['velocities']
        
        avg_speed = np.mean(np.linalg.norm(velocities, axis=2))
        max_altitude = np.max(trajectories[:, :, 2])
        min_altitude = np.min(trajectories[:, :, 2])
        
        print(f"  Average speed: {avg_speed:.2f} m/s")
        print(f"  Altitude range: {min_altitude:.1f} - {max_altitude:.1f} m")
        print(f"  Trajectory shape: {trajectories.shape}")
        
        # Visualize first scenario
        if pattern == FlightPattern.COORDINATED_ATTACK:
            fig = visualize_trajectories(data)
            plt.show()
            
    print("\nTrajectory generation complete!")