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
    
    # Every drone holds a cruise altitude above ground drawn from this range
    # (meters AGL). Patterns differ only in their horizontal paths, so coverage
    # and ground resolution are compared at the same flying height.
    CRUISE_AGL_RANGE = (150.0, 250.0)

    def __init__(self, bounds: SimulationBounds, lane_spacing_m: float = 200.0):
        """
        Args:
            bounds: simulation volume in REAL meters (x east, y north, z AGL).
                    Pattern geometry (start points, center, ring radius) is
                    defined as fractions of these bounds; spacings, speeds
                    and accelerations are physical meters and m/s.
            lane_spacing_m: lateral offset applied when a sweeping drone turns
                    at a boundary (boustrophedon lane change). Should be about
                    one camera swath width.
        """
        self.bounds = bounds
        self.lane_spacing_m = lane_spacing_m
        self._width  = bounds.x_max - bounds.x_min
        self._height = bounds.y_max - bounds.y_min
        # Per-drone pattern state (sweep direction, lane target), reset per swarm
        self._drone_state = {}
        
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

        # Clear per-swarm state so a given random seed always yields the same
        # trajectories, regardless of what was generated earlier.
        self._drone_state = {}
        if hasattr(self, '_evasive_direction'):
            del self._evasive_direction
        self._num_drones = num_drones
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
        b = self.bounds
        cx, cy = self._center()

        if pattern == FlightPattern.COORDINATED_ATTACK:
            # Start from a ring at 40% of the shorter side, converge to center
            # (originally a 400 m ring in a 1 km box)
            angles = np.linspace(0, 2*np.pi, num_drones, endpoint=False)
            radius = 0.4 * min(self._width, self._height)

            positions[:, 0] = cx + radius * np.cos(angles)
            positions[:, 1] = cy + radius * np.sin(angles)
            positions[:, 2] = np.random.uniform(*self.CRUISE_AGL_RANGE, num_drones)

        elif pattern == FlightPattern.FORMATION_FLYING:
            # V-formation or grid formation (physical spacing in meters)
            if num_drones <= 10:
                # V-formation
                positions = self._generate_v_formation(num_drones)
            else:
                # Grid formation
                positions = self._generate_grid_formation(num_drones)

        elif pattern == FlightPattern.RANDOM_DISPERSAL:
            # Random starting positions
            positions[:, 0] = np.random.uniform(b.x_min + 50, b.x_max - 50, num_drones)
            positions[:, 1] = np.random.uniform(b.y_min + 50, b.y_max - 50, num_drones)
            positions[:, 2] = np.random.uniform(*self.CRUISE_AGL_RANGE, num_drones)

        elif pattern == FlightPattern.EVASIVE_MANEUVERS:
            # Start clustered (50 m radius) at 20% / 20% of the area, then spread out
            center = np.array([b.x_min + 0.2 * self._width,
                               b.y_min + 0.2 * self._height, 200.0])
            cluster_radius = 50.0
            positions = center + np.random.normal(0, cluster_radius, (num_drones, 3))
            positions[:, 2] = np.random.uniform(*self.CRUISE_AGL_RANGE, num_drones)

        elif pattern == FlightPattern.PERIMETER_SWEEP:
            # Start along the west edge, one lane per drone
            positions[:, 0] = b.x_min + 20
            positions[:, 1] = np.linspace(b.y_min + 50, b.y_max - 50, num_drones)
            positions[:, 2] = np.random.uniform(*self.CRUISE_AGL_RANGE, num_drones)

        if pattern == FlightPattern.FORMATION_FLYING:
            # Formation helpers set ~200 m with small jitter; use the common range
            positions[:, 2] = np.random.uniform(*self.CRUISE_AGL_RANGE, num_drones)

        return positions

    def _center(self) -> Tuple[float, float]:
        return ((self.bounds.x_min + self.bounds.x_max) / 2,
                (self.bounds.y_min + self.bounds.y_max) / 2)

    def _turn_margin(self, spec: DroneSpec) -> float:
        """Distance from an x-boundary at which a sweeping drone starts its turn."""
        return 0.5 * spec.max_velocity ** 2 / spec.max_acceleration + 50.0

    def _sweep_state(self, drone_id: int, pos: np.ndarray) -> dict:
        """Per-drone sweep direction and lane target, created on first use."""
        st = self._drone_state.get(drone_id)
        if st is None:
            n = max(getattr(self, '_num_drones', 1), 1)
            strip = (self._height - 100.0) / n   # lateral strip owned by this drone
            # Keep the strip inside the area: a drone starting near the north or
            # south edge would otherwise own a strip that runs off the map, and its
            # lane changes got clipped to a ~40 m band instead of ~200 m.
            lo = max(float(pos[1]) - strip / 2, self.bounds.y_min + 20)
            hi = min(float(pos[1]) + strip / 2, self.bounds.y_max - 20)
            st = {'dir': 1.0, 'lane_y': float(pos[1]), 'lane_step': 1.0,
                  'strip_lo': lo, 'strip_hi': hi}
            self._drone_state[drone_id] = st
        return st

    def _update_sweep_direction(self, st: dict, pos: np.ndarray, spec: DroneSpec,
                                lane_change: bool) -> None:
        """Reverse the x direction near the east/west boundary instead of
        piling up against it; optionally move to the next lane (boustrophedon)."""
        m = self._turn_margin(spec)
        turned = False
        if st['dir'] > 0 and pos[0] >= self.bounds.x_max - m:
            st['dir'] = -1.0; turned = True
        elif st['dir'] < 0 and pos[0] <= self.bounds.x_min + m:
            st['dir'] = 1.0; turned = True
        if turned and lane_change:
            nxt = st['lane_y'] + st['lane_step'] * self.lane_spacing_m
            if nxt > st['strip_hi'] or nxt < st['strip_lo']:
                st['lane_step'] *= -1.0
                nxt = st['lane_y'] + st['lane_step'] * self.lane_spacing_m
            st['lane_y'] = float(np.clip(nxt, self.bounds.y_min + 20, self.bounds.y_max - 20))
    
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
            # Altitude hold: mapping autopilots fly at a set height, so no
            # pattern maneuvers vertically. (Previously the dispersal patterns
            # wandered across 60-490 m, so their coverage and resolution
            # reflected altitude as well as flight path.)
            desired_vel[2] = 0.0
            
            # Apply physical constraints
            desired_vel = self._apply_velocity_constraints(desired_vel, current_vel, spec, dt)
            
            # Add environmental effects
            # Partial wind effect, horizontal only: the autopilot holds
            # altitude. (Adding the 0.5 m/s vertical component here made every
            # drone climb a steady 0.05 m/s, about +90 m per 30 minutes, which
            # widened camera footprints later in each flight.)
            wind_effect = self.wind_velocity * 0.1
            wind_effect[2] = 0.0
            ground_vel = desired_vel + wind_effect

            # Update position with the ground velocity...
            current_pos += ground_vel * dt
            # ...but carry forward only the air-relative velocity the autopilot
            # controls. (Carrying the ground velocity let the wind feed back
            # through the momentum term of random dispersal and compound to a
            # ~4.5 m/s drift instead of the intended 0.2 m/s.)
            current_vel = desired_vel.copy()

            # Apply boundary constraints (reflect instead of clamping, so drones
            # turn around rather than accumulating against the boundary)
            current_pos, current_vel = self._apply_boundary_constraints(current_pos, current_vel)

            # Add sensor noise
            noisy_pos = current_pos + np.random.normal(0, self.noise_std, 3)

            trajectory[i] = noisy_pos
            velocity[i] = current_vel + wind_effect   # ground velocity
            
        return trajectory, velocity
    
    def _calculate_desired_velocity(self, drone_id: int, pos: np.ndarray, 
                                  vel: np.ndarray, t: float, 
                                  pattern: FlightPattern, spec: DroneSpec) -> np.ndarray:
        """Calculate desired velocity based on flight pattern"""
        
        if pattern == FlightPattern.COORDINATED_ATTACK:
            # Move toward center with slight spiral
            cx, cy = self._center()
            center = np.array([cx, cy, 200.0])
            to_center = center - pos
            distance = np.linalg.norm(to_center)

            if distance > 50:
                # Spiral component. Originally 0.1 * distance in a 1 km box; the
                # distance is expressed in those normalized units so the spiral
                # keeps the same shape at real scale.
                norm = 1000.0 / min(self._width, self._height)
                spiral_vel = np.array([-to_center[1], to_center[0], 0]) * 0.1 * norm
                desired = (to_center / distance) * spec.max_velocity * 0.7 + spiral_vel
            else:
                # Hover near center
                desired = np.random.normal(0, 2.0, 3)
                
        elif pattern == FlightPattern.FORMATION_FLYING:
            # Maintain formation while moving forward; turn back at the boundary
            st = self._sweep_state(drone_id, pos)
            self._update_sweep_direction(st, pos, spec, lane_change=False)
            formation_vel = np.array([st['dir'] * spec.max_velocity * 0.6, 0, 0])
            
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
                self._evasive_direction[2] = 0.0          # horizontal heading
                self._evasive_direction = self._evasive_direction / np.linalg.norm(self._evasive_direction)
            
            if not hasattr(self, '_evasive_direction'):
                self._evasive_direction = np.array([1, 0, 0])
                
            base_vel = self._evasive_direction * spec.max_velocity * 0.8
            zigzag = np.array([0, np.sin(t * 3) * 5, np.cos(t * 2) * 3])
            desired = base_vel + zigzag
            
        elif pattern == FlightPattern.PERIMETER_SWEEP:
            # Sweep across the area; at each boundary turn back and shift one
            # lane within this drone's strip (boustrophedon)
            st = self._sweep_state(drone_id, pos)
            self._update_sweep_direction(st, pos, spec, lane_change=True)
            sweep_vel = np.array([st['dir'] * spec.max_velocity * 0.5, 0, 0])

            # Steer laterally toward the current lane
            lane_vel = np.clip(0.2 * (st['lane_y'] - pos[1]),
                               -0.3 * spec.max_velocity, 0.3 * spec.max_velocity)

            # Add slight weaving
            weave = np.array([0, np.sin(t + drone_id) * 3 + lane_vel, 0])
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
    
    def _apply_boundary_constraints(self, pos: np.ndarray,
                                    vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Keep drones within simulation boundaries by reflecting off them.

        The original version clamped positions, which made drones accumulate
        on the boundary (and produced an artificial coverage plateau)."""
        lo = np.array([self.bounds.x_min, self.bounds.y_min, self.bounds.z_min]) + 10
        hi = np.array([self.bounds.x_max, self.bounds.y_max, self.bounds.z_max]) - 10
        for k in range(3):
            if pos[k] < lo[k]:
                pos[k] = min(2 * lo[k] - pos[k], hi[k]); vel[k] = abs(vel[k])
            elif pos[k] > hi[k]:
                pos[k] = max(2 * hi[k] - pos[k], lo[k]); vel[k] = -abs(vel[k])
        return pos, vel
    
    def _calculate_formation_forces(self, drone_id: int, pos: np.ndarray) -> np.ndarray:
        """Calculate forces to maintain formation (simplified)"""
        # This is a placeholder - you can extend this for complex formations
        return np.random.normal(0, 1.0, 3)
    
    def _generate_v_formation(self, num_drones: int) -> np.ndarray:
        """Generate V-formation starting positions"""
        positions = np.zeros((num_drones, 3))
        
        # Leader at front (20% across, mid-height; spacing in physical meters)
        lx = self.bounds.x_min + 0.2 * self._width
        ly = self._center()[1]
        positions[0] = [lx, ly, 200]

        # Followers in V shape
        for i in range(1, num_drones):
            side = 1 if i % 2 == 0 else -1  # Alternate sides
            row = (i + 1) // 2

            positions[i] = [
                lx - row * 20,  # Move back
                ly + side * row * 15,  # Move to side
                200 + np.random.uniform(-10, 10)  # Small altitude variation
            ]
            
        return positions
    
    def _generate_grid_formation(self, num_drones: int) -> np.ndarray:
        """Generate grid formation starting positions"""
        grid_size = int(np.ceil(np.sqrt(num_drones)))
        positions = np.zeros((num_drones, 3))
        
        spacing = 30.0  # meters between drones
        start_x = self.bounds.x_min + 0.2 * self._width
        start_y = self.bounds.y_min + 0.2 * self._height
        
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