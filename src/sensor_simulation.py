"""
Virtual Sensor Array Simulation for Multi-Drone Detection
Simulates multi-camera sensor arrays with realistic limitations and noise
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Union
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count
import time
import os

class SensorType(Enum):
    VISIBLE_SPECTRUM = "visible"
    THERMAL_INFRARED = "thermal"
    RADAR = "radar"
    LIDAR = "lidar"

@dataclass
class CameraSpecification:
    """Camera sensor specifications and limitations"""
    sensor_type: SensorType
    resolution: Tuple[int, int]  # (width, height) in pixels
    field_of_view: Tuple[float, float]  # (horizontal, vertical) in degrees
    max_range: float  # Maximum detection range in meters
    min_range: float  # Minimum detection range in meters
    noise_std: float  # Sensor noise standard deviation
    detection_threshold: float  # Minimum object size for detection (meters)
    frame_rate: float  # Frames per second
    angular_resolution: float  # Angular resolution in degrees

@dataclass
class CameraPosition:
    """3D position and orientation of a camera sensor"""
    position: np.ndarray  # [x, y, z] in meters
    orientation: np.ndarray  # [yaw, pitch, roll] in degrees
    camera_id: str
    spec: CameraSpecification

@dataclass
class SensorObservation:
    """Single sensor observation of detected targets"""
    camera_id: str
    timestamp: float
    detected_objects: List[Dict]  # List of detected object data
    image_coordinates: np.ndarray  # 2D pixel coordinates of detections
    confidence_scores: np.ndarray  # Detection confidence [0-1]
    range_estimates: np.ndarray  # Estimated ranges to targets
    sensor_noise_level: float  # Current noise level

class VirtualSensorArray:
    """
    Simulates a multi-camera sensor array for drone swarm detection.
    Includes realistic sensor limitations, noise, and detection capabilities.
    """
    
    def __init__(self, simulation_bounds):
        self.bounds = simulation_bounds
        self.cameras = []
        self.current_time = 0.0
        
        # Environmental conditions
        self.weather_conditions = {
            'visibility': 1.0,  # 1.0 = clear, 0.0 = completely obscured
            'wind_speed': 5.0,  # m/s
            'temperature': 20.0,  # Celsius
            'humidity': 0.6  # 0-1
        }
        
        # Detection history for tracking
        self.detection_history = []
        
    def add_camera(self, position: np.ndarray, orientation: np.ndarray, 
                   spec: CameraSpecification, camera_id: str) -> None:
        """Add a camera to the sensor array"""
        camera = CameraPosition(
            position=np.array(position),
            orientation=np.array(orientation),
            camera_id=camera_id,
            spec=spec
        )
        self.cameras.append(camera)
        
    def setup_perimeter_array(self, num_cameras: int = 8, 
                            sensor_type: SensorType = SensorType.VISIBLE_SPECTRUM) -> None:
        """Set up cameras in a perimeter configuration around the surveillance area"""
        
        # Define camera specifications based on type
        if sensor_type == SensorType.VISIBLE_SPECTRUM:
            spec = CameraSpecification(
                sensor_type=sensor_type,
                resolution=(1920, 1080),
                field_of_view=(60.0, 45.0),  # degrees
                max_range=2000.0,  # meters
                min_range=50.0,
                noise_std=2.0,  # pixels
                detection_threshold=0.3,  # meters
                frame_rate=30.0,
                angular_resolution=0.05  # degrees
            )
        elif sensor_type == SensorType.THERMAL_INFRARED:
            spec = CameraSpecification(
                sensor_type=sensor_type,
                resolution=(640, 480),
                field_of_view=(45.0, 35.0),
                max_range=1500.0,
                min_range=30.0,
                noise_std=1.5,
                detection_threshold=0.4,
                frame_rate=25.0,
                angular_resolution=0.08
            )
        else:
            # Default visible spectrum
            spec = CameraSpecification(
                sensor_type=sensor_type,
                resolution=(1280, 720),
                field_of_view=(50.0, 40.0),
                max_range=1800.0,
                min_range=40.0,
                noise_std=2.5,
                detection_threshold=0.35,
                frame_rate=25.0,
                angular_resolution=0.06
            )
        
        # Position cameras around perimeter
        center_x = (self.bounds.x_max + self.bounds.x_min) / 2
        center_y = (self.bounds.y_max + self.bounds.y_min) / 2
        radius = min(self.bounds.x_max - center_x, self.bounds.y_max - center_y) * 1.2
        
        for i in range(num_cameras):
            angle = 2 * np.pi * i / num_cameras
            
            # Camera position (outside the surveillance area)
            cam_x = center_x + radius * np.cos(angle)
            cam_y = center_y + radius * np.sin(angle)
            cam_z = 100.0  # 100m elevation
            
            # Camera orientation (looking toward center)
            yaw = np.degrees(angle + np.pi)  # Look toward center
            pitch = -15.0  # Slight downward angle
            roll = 0.0
            
            self.add_camera(
                position=[cam_x, cam_y, cam_z],
                orientation=[yaw, pitch, roll],
                spec=spec,
                camera_id=f"{sensor_type.value}_cam_{i:02d}"
            )
    
    def setup_triangulation_array(self, baseline_distance: float = 300.0,
                                sensor_type: SensorType = SensorType.VISIBLE_SPECTRUM) -> None:
        """Set up optimized camera array for triangulation accuracy"""
        
        spec = CameraSpecification(
            sensor_type=sensor_type,
            resolution=(2048, 1536),  # Higher resolution for accuracy
            field_of_view=(65.0, 50.0),
            max_range=2500.0,
            min_range=30.0,
            noise_std=1.0,  # Lower noise for precision
            detection_threshold=0.25,
            frame_rate=30.0,
            angular_resolution=0.03
        )
        
        center_x = (self.bounds.x_max + self.bounds.x_min) / 2
        center_y = (self.bounds.y_max + self.bounds.y_min) / 2
        
        # Optimal camera positions for triangulation
        positions = [
            [center_x - baseline_distance, center_y - baseline_distance, 150],  # SW
            [center_x + baseline_distance, center_y - baseline_distance, 150],  # SE
            [center_x + baseline_distance, center_y + baseline_distance, 150],  # NE
            [center_x - baseline_distance, center_y + baseline_distance, 150],  # NW
            [center_x, center_y - baseline_distance * 1.5, 200],               # S (elevated)
            [center_x, center_y + baseline_distance * 1.5, 200]                # N (elevated)
        ]
        
        for i, pos in enumerate(positions):
            # Calculate orientation toward surveillance center
            dx = center_x - pos[0]
            dy = center_y - pos[1]
            yaw = np.degrees(np.arctan2(dy, dx))
            pitch = -10.0  # Slight downward angle
            
            self.add_camera(
                position=pos,
                orientation=[yaw, pitch, 0.0],
                spec=spec,
                camera_id=f"triangulation_cam_{i:02d}"
            )
    
    def observe_targets(self, drone_positions: np.ndarray, 
                       timestamp: float) -> List[SensorObservation]:
        """
        Generate sensor observations for given drone positions
        
        Args:
            drone_positions: Array of shape (num_drones, 3) with [x, y, z] positions
            timestamp: Current simulation time
            
        Returns:
            List of sensor observations from each camera
        """
        observations = []
        self.current_time = timestamp
        
        for camera in self.cameras:
            observation = self._simulate_camera_observation(camera, drone_positions, timestamp)
            observations.append(observation)
            
        return observations
    
    def observe_targets_parallel(self, drone_positions: np.ndarray, 
                                timestamp: float, max_workers: int = None) -> List[SensorObservation]:
        """
        Parallel version of observe_targets - processes cameras simultaneously

        Args:
            drone_positions: Array of shape (num_drones, 3) with [x, y, z] positions
            timestamp: Current simulation time
            max_workers: Maximum number of worker threads (default: auto-detect)

        Returns:
            List of sensor observations from each camera (same format as observe_targets)
        """
        if max_workers is None:
            max_workers = min(cpu_count(), len(self.cameras), 8)

        self.current_time = timestamp

        def process_camera(camera):
            """Process a single camera - same logic as the original"""
            return self._simulate_camera_observation(camera, drone_positions, timestamp)

        # Process all cameras in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_camera, camera) for camera in self.cameras]
            observations = [future.result() for future in futures]

        return observations


    def _simulate_camera_observation(self, camera: CameraPosition, 
                                   drone_positions: np.ndarray,
                                   timestamp: float) -> SensorObservation:
        """Simulate observation from a single camera"""
        
        detected_objects = []
        image_coordinates = []
        confidence_scores = []
        range_estimates = []
        
        for i, drone_pos in enumerate(drone_positions):
            # Check if drone is within sensor range and field of view
            detection_result = self._check_detection_feasibility(camera, drone_pos)
            
            if detection_result['detectable']:
                # Calculate image coordinates
                img_coords = self._world_to_image_coordinates(camera, drone_pos)
                
                # Add sensor noise
                noisy_coords = self._add_sensor_noise(camera, img_coords)
                
                # Calculate range estimate with noise
                true_range = detection_result['range']
                range_noise = np.random.normal(0, true_range * 0.02)  # 2% range error
                estimated_range = true_range + range_noise
                
                # Calculate confidence based on range, size, and conditions
                confidence = self._calculate_detection_confidence(camera, drone_pos, detection_result)
                
                detected_objects.append({
                    'drone_id': i,
                    'world_position': drone_pos,
                    'estimated_size': detection_result['apparent_size'],
                    'detection_quality': detection_result['quality']
                })
                
                image_coordinates.append(noisy_coords)
                confidence_scores.append(confidence)
                range_estimates.append(estimated_range)
        
        # Convert to numpy arrays
        image_coordinates = np.array(image_coordinates) if image_coordinates else np.empty((0, 2))
        confidence_scores = np.array(confidence_scores) if confidence_scores else np.empty(0)
        range_estimates = np.array(range_estimates) if range_estimates else np.empty(0)
        
        # Calculate current sensor noise level based on conditions
        noise_level = self._calculate_environmental_noise_factor(camera)
        
        return SensorObservation(
            camera_id=camera.camera_id,
            timestamp=timestamp,
            detected_objects=detected_objects,
            image_coordinates=image_coordinates,
            confidence_scores=confidence_scores,
            range_estimates=range_estimates,
            sensor_noise_level=noise_level
        )
    
    def _check_detection_feasibility(self, camera: CameraPosition, 
                                   drone_pos: np.ndarray) -> Dict:
        """Check if a drone can be detected by the camera"""
        
        # Calculate relative position
        relative_pos = drone_pos - camera.position
        range_to_target = np.linalg.norm(relative_pos)
        
        # Range check
        if range_to_target < camera.spec.min_range or range_to_target > camera.spec.max_range:
            return {'detectable': False, 'reason': 'out_of_range'}
        
        # Field of view check
        if not self._is_in_field_of_view(camera, drone_pos):
            return {'detectable': False, 'reason': 'outside_fov'}
        
        # Calculate apparent size
        drone_size = 0.8  # Assume 0.8m wingspan
        apparent_size = drone_size / range_to_target
        
        # Size threshold check
        if apparent_size < camera.spec.detection_threshold / range_to_target:
            return {'detectable': False, 'reason': 'too_small'}
        
        # Environmental effects
        visibility_factor = self._calculate_visibility_factor(range_to_target)
        if visibility_factor < 0.3:  # 30% minimum visibility
            return {'detectable': False, 'reason': 'poor_visibility'}
        
        # Detection quality based on multiple factors
        quality = min(1.0, apparent_size * 100) * visibility_factor
        
        return {
            'detectable': True,
            'range': range_to_target,
            'apparent_size': apparent_size,
            'quality': quality,
            'visibility_factor': visibility_factor
        }
    
    def _is_in_field_of_view(self, camera: CameraPosition, target_pos: np.ndarray) -> bool:
        """Check if target is within camera's field of view"""
        
        # Transform to camera coordinate system
        relative_pos = target_pos - camera.position
        
        # Apply camera rotation (simplified - assumes only yaw rotation for now)
        yaw_rad = np.radians(camera.orientation[0])
        cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
        
        # Rotate to camera frame
        cam_x = relative_pos[0] * cos_yaw + relative_pos[1] * sin_yaw
        cam_y = -relative_pos[0] * sin_yaw + relative_pos[1] * cos_yaw
        cam_z = relative_pos[2]
        
        # Check if in front of camera
        if cam_x <= 0:
            return False
        
        # Calculate angles
        horizontal_angle = np.degrees(np.arctan2(cam_y, cam_x))
        vertical_angle = np.degrees(np.arctan2(cam_z, cam_x))
        
        # Check field of view bounds
        h_fov, v_fov = camera.spec.field_of_view
        
        return (abs(horizontal_angle) <= h_fov/2 and 
                abs(vertical_angle) <= v_fov/2)
    
    def _world_to_image_coordinates(self, camera: CameraPosition, 
                                  world_pos: np.ndarray) -> np.ndarray:
        """Convert world coordinates to image pixel coordinates"""
        
        # Simplified projection model
        relative_pos = world_pos - camera.position
        range_to_target = np.linalg.norm(relative_pos)
        
        # Apply camera rotation
        yaw_rad = np.radians(camera.orientation[0])
        cos_yaw, sin_yaw = np.cos(yaw_rad), np.sin(yaw_rad)
        
        cam_x = relative_pos[0] * cos_yaw + relative_pos[1] * sin_yaw
        cam_y = -relative_pos[0] * sin_yaw + relative_pos[1] * cos_yaw
        cam_z = relative_pos[2]
        
        # Project to image plane
        if cam_x > 0:  # In front of camera
            h_angle = np.arctan2(cam_y, cam_x)
            v_angle = np.arctan2(cam_z, cam_x)
            
            # Convert to pixel coordinates
            h_fov_rad = np.radians(camera.spec.field_of_view[0])
            v_fov_rad = np.radians(camera.spec.field_of_view[1])
            
            # Normalize to [-1, 1] then to pixel coordinates
            norm_x = h_angle / (h_fov_rad / 2)
            norm_y = v_angle / (v_fov_rad / 2)
            
            pixel_x = (norm_x + 1) * camera.spec.resolution[0] / 2
            pixel_y = (1 - norm_y) * camera.spec.resolution[1] / 2  # Flip Y axis
            
            return np.array([pixel_x, pixel_y])
        
        return np.array([0, 0])  # Default for behind camera
    
    def _add_sensor_noise(self, camera: CameraPosition, 
                         image_coords: np.ndarray) -> np.ndarray:
        """Add realistic sensor noise to image coordinates"""
        
        # Environmental noise factor
        env_factor = self._calculate_environmental_noise_factor(camera)
        
        # Base sensor noise
        base_noise = camera.spec.noise_std
        
        # Total noise (environmental effects increase noise)
        total_noise = base_noise * (1.0 + env_factor)
        
        # Add Gaussian noise
        noise = np.random.normal(0, total_noise, 2)
        
        return image_coords + noise
    
    def _calculate_detection_confidence(self, camera: CameraPosition,
                                      drone_pos: np.ndarray,
                                      detection_info: Dict) -> float:
        """Calculate detection confidence score [0-1]"""
        
        base_confidence = 0.8  # Base confidence for good conditions
        
        # Range factor (closer is better)
        range_factor = 1.0 - (detection_info['range'] / camera.spec.max_range)
        
        # Size factor (larger apparent size is better)
        size_factor = min(1.0, detection_info['apparent_size'] * 500)
        
        # Environmental factor
        env_factor = detection_info['visibility_factor']
        
        # Sensor quality factor
        quality_factor = detection_info['quality']
        
        # Combine factors
        confidence = (base_confidence * 
                     (0.3 * range_factor + 
                      0.3 * size_factor + 
                      0.2 * env_factor + 
                      0.2 * quality_factor))
        
        # Add some random variation
        confidence += np.random.normal(0, 0.05)
        
        return np.clip(confidence, 0.1, 0.95)  # Keep within reasonable bounds
    
    def _calculate_visibility_factor(self, range_to_target: float) -> float:
        """Calculate visibility factor based on environmental conditions and range"""
        
        base_visibility = self.weather_conditions['visibility']
        
        # Atmospheric attenuation with distance
        attenuation = np.exp(-range_to_target / 5000.0)  # 5km characteristic distance
        
        # Weather effects
        humidity_effect = 1.0 - 0.3 * self.weather_conditions['humidity']
        
        return base_visibility * attenuation * humidity_effect
    
    def _calculate_environmental_noise_factor(self, camera: CameraPosition) -> float:
        """Calculate environmental noise factor"""
        
        base_factor = 0.1
        
        # Weather effects
        visibility_effect = (1.0 - self.weather_conditions['visibility']) * 0.5
        wind_effect = min(0.3, self.weather_conditions['wind_speed'] / 20.0)
        
        # Time of day effect (simplified)
        time_effect = 0.1 * np.sin(self.current_time / 86400 * 2 * np.pi) ** 2
        
        return base_factor + visibility_effect + wind_effect + time_effect
    
    def get_detection_statistics(self, observations: List[SensorObservation]) -> Dict:
        """Calculate detection statistics across all sensors"""
        
        total_detections = sum(len(obs.detected_objects) for obs in observations)
        active_cameras = len([obs for obs in observations if len(obs.detected_objects) > 0])
        
        if total_detections > 0:
            avg_confidence = np.mean([np.mean(obs.confidence_scores) 
                                    for obs in observations 
                                    if len(obs.confidence_scores) > 0])
            avg_range = np.mean([np.mean(obs.range_estimates) 
                               for obs in observations 
                               if len(obs.range_estimates) > 0])
        else:
            avg_confidence = 0.0
            avg_range = 0.0
        
        return {
            'total_detections': total_detections,
            'active_cameras': active_cameras,
            'total_cameras': len(self.cameras),
            'detection_density': total_detections / len(self.cameras) if self.cameras else 0,
            'average_confidence': avg_confidence,
            'average_range': avg_range,
            'timestamp': self.current_time
        }
    
    def visualize_sensor_coverage(self, elevation_slice: float = 200.0) -> plt.Figure:
        """Visualize sensor array coverage at specified elevation - FIXED VERSION"""
        
        # Larger figure with better proportions
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Calculate extended bounds to include all cameras
        all_x = [cam.position[0] for cam in self.cameras] + [self.bounds.x_min, self.bounds.x_max]
        all_y = [cam.position[1] for cam in self.cameras] + [self.bounds.y_min, self.bounds.y_max]
        
        x_margin = (max(all_x) - min(all_x)) * 0.15  # 15% margin
        y_margin = (max(all_y) - min(all_y)) * 0.15
        
        ax1.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax1.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        
        print(f"DEBUG: Displaying {len(self.cameras)} cameras")
        print(f"Camera positions: {[(cam.position[0], cam.position[1]) for cam in self.cameras]}")
        print(f"Plot bounds: X=[{min(all_x)-x_margin:.0f}, {max(all_x)+x_margin:.0f}], Y=[{min(all_y)-y_margin:.0f}, {max(all_y)+y_margin:.0f}]")
        
        for i, camera in enumerate(self.cameras):
            # Larger camera markers
            ax1.plot(camera.position[0], camera.position[1], 'ro', markersize=12, 
                    markeredgecolor='black', markeredgewidth=1.5)
            
            # Better camera labels with background
            ax1.annotate(f'Cam{i+1}', 
                        (camera.position[0], camera.position[1]),
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=11, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.8),
                        ha='left', va='bottom')
            
            # Draw field of view cone (simplified 2D projection)
            yaw = np.radians(camera.orientation[0])
            h_fov = np.radians(camera.spec.field_of_view[0])
            fov_range = camera.spec.max_range * 0.4  # Shorter lines for clarity
            
            # FOV boundary rays
            angles = [yaw - h_fov/2, yaw + h_fov/2]
            for angle in angles:
                end_x = camera.position[0] + fov_range * np.cos(angle)
                end_y = camera.position[1] + fov_range * np.sin(angle)
                ax1.plot([camera.position[0], end_x], 
                        [camera.position[1], end_y], 'b--', alpha=0.7, linewidth=2)
        
        # Draw surveillance area boundary as a rectangle
        rect = plt.Rectangle((self.bounds.x_min, self.bounds.y_min), 
                            self.bounds.x_max - self.bounds.x_min,
                            self.bounds.y_max - self.bounds.y_min,
                            fill=False, edgecolor='green', linewidth=3, label='Surveillance Area')
        ax1.add_patch(rect)
        
        ax1.set_xlabel('X (meters)', fontsize=12)
        ax1.set_ylabel('Y (meters)', fontsize=12)
        ax1.set_title(f'Sensor Array Layout ({len(self.cameras)} cameras)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        ax1.legend()
        
        # Plot 2: Detection range vs angle for first camera
        if self.cameras:
            camera = self.cameras[0]
            angles = np.linspace(-90, 90, 181)
            detection_ranges = []
            
            for angle in angles:
                # Test detection at this angle
                test_range = camera.spec.max_range * 0.8
                test_x = camera.position[0] + test_range * np.cos(np.radians(angle))
                test_y = camera.position[1] + test_range * np.sin(np.radians(angle))
                test_pos = np.array([test_x, test_y, elevation_slice])
                
                detection = self._check_detection_feasibility(camera, test_pos)
                if detection['detectable']:
                    detection_ranges.append(detection['range'])
                else:
                    detection_ranges.append(0)
            
            ax2.plot(angles, detection_ranges, 'g-', linewidth=2)
            ax2.fill_between(angles, detection_ranges, alpha=0.3, color='green')
            ax2.set_xlabel('Angle (degrees)', fontsize=12)
            ax2.set_ylabel('Detection Range (meters)', fontsize=12)
            ax2.set_title(f'Detection Range Profile - {camera.camera_id}', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # Better spacing to prevent clipping - THIS IS THE KEY FIX
        plt.tight_layout(pad=2.0)
        plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.12, wspace=0.3)
        
        return fig

    # NEW DETECTION RANGE ANALYSIS METHODS - ADD THESE FOR THE NEW FEATURES
    def visualize_all_camera_detection_ranges(self, elevation_slice: float = 200.0, output_dir: str = 'images') -> List[str]:
        """
        Generate separate detection range plots for each camera
        Returns list of saved file paths
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_files = []
        
        print(f"Generating detection range plots for {len(self.cameras)} cameras...")
        
        for i, camera in enumerate(self.cameras):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Left plot: Camera position in context
            self._plot_camera_context(ax1, camera, i)
            
            # Right plot: Detection range profile
            self._plot_single_camera_detection_range(ax2, camera, elevation_slice)
            
            # Save individual camera plot
            filename = f'camera_{i+1:02d}_detection_range.png'
            filepath = os.path.join(output_dir, filename)
            
            plt.tight_layout(pad=2.0)
            plt.subplots_adjust(left=0.08, right=0.96, top=0.92, bottom=0.12, wspace=0.3)
            
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            saved_files.append(filepath)
            print(f"  Saved: {filename}")
        
        # Also create a summary comparison plot
        summary_file = self.create_detection_range_summary(elevation_slice, output_dir)
        saved_files.append(summary_file)
        
        return saved_files

    def _plot_camera_context(self, ax, target_camera, camera_index):
        """Plot the camera's position in context of the full array"""
        
        # Calculate bounds
        all_x = [cam.position[0] for cam in self.cameras] + [self.bounds.x_min, self.bounds.x_max]
        all_y = [cam.position[1] for cam in self.cameras] + [self.bounds.y_min, self.bounds.y_max]
        
        x_margin = (max(all_x) - min(all_x)) * 0.15
        y_margin = (max(all_y) - min(all_y)) * 0.15
        
        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        
        # Plot all cameras (dimmed)
        for i, camera in enumerate(self.cameras):
            if i == camera_index:
                # Highlight the target camera
                ax.plot(camera.position[0], camera.position[1], 'ro', 
                       markersize=15, markeredgecolor='black', markeredgewidth=2,
                       label=f'Camera {i+1} (Active)')
                
                # Show its FOV in detail
                yaw = np.radians(camera.orientation[0])
                h_fov = np.radians(camera.spec.field_of_view[0])
                fov_range = camera.spec.max_range * 0.4
                
                # FOV boundary lines
                for angle in [yaw - h_fov/2, yaw + h_fov/2]:
                    end_x = camera.position[0] + fov_range * np.cos(angle)
                    end_y = camera.position[1] + fov_range * np.sin(angle)
                    ax.plot([camera.position[0], end_x], [camera.position[1], end_y], 
                           'r-', alpha=0.8, linewidth=3)
                
                # FOV arc
                arc_angles = np.linspace(yaw - h_fov/2, yaw + h_fov/2, 20)
                arc_x = camera.position[0] + fov_range * np.cos(arc_angles)
                arc_y = camera.position[1] + fov_range * np.sin(arc_angles)
                ax.plot(arc_x, arc_y, 'r-', alpha=0.6, linewidth=2)
                
            else:
                # Other cameras (dimmed)
                ax.plot(camera.position[0], camera.position[1], 'gray', marker='o', 
                       markersize=8, alpha=0.5)
                ax.annotate(f'{i+1}', (camera.position[0], camera.position[1]),
                           xytext=(0, 0), textcoords='offset points',
                           fontsize=8, ha='center', va='center', alpha=0.6)
        
        # Surveillance area
        rect = plt.Rectangle((self.bounds.x_min, self.bounds.y_min), 
                            self.bounds.x_max - self.bounds.x_min,
                            self.bounds.y_max - self.bounds.y_min,
                            fill=False, edgecolor='green', linewidth=2, alpha=0.7)
        ax.add_patch(rect)
        
        ax.set_xlabel('X (meters)', fontsize=11)
        ax.set_ylabel('Y (meters)', fontsize=11)
        ax.set_title(f'Camera {camera_index+1} Position & Field of View', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend()

    def _plot_single_camera_detection_range(self, ax, camera, elevation_slice):
        """Plot detection range profile for a single camera"""
        
        angles = np.linspace(-90, 90, 181)
        detection_ranges = []
        quality_scores = []
        
        for angle in angles:
            # Test detection at this angle relative to camera orientation
            test_range = camera.spec.max_range * 0.8
            actual_angle = np.radians(angle + camera.orientation[0])
            test_x = camera.position[0] + test_range * np.cos(actual_angle)
            test_y = camera.position[1] + test_range * np.sin(actual_angle)
            test_pos = np.array([test_x, test_y, elevation_slice])
            
            detection = self._check_detection_feasibility(camera, test_pos)
            if detection['detectable']:
                detection_ranges.append(detection['range'])
                quality_scores.append(detection.get('quality', 0.5))
            else:
                detection_ranges.append(0)
                quality_scores.append(0)
        
        # Main detection range plot
        ax.plot(angles, detection_ranges, 'b-', linewidth=3, label='Detection Range')
        ax.fill_between(angles, detection_ranges, alpha=0.3, color='blue')
        
        # Add quality overlay
        ax2 = ax.twinx()
        ax2.plot(angles, quality_scores, 'orange', linewidth=2, alpha=0.7, label='Detection Quality')
        ax2.set_ylabel('Detection Quality', color='orange', fontsize=11)
        ax2.set_ylim(0, 1)
        
        # Calculate and display statistics
        max_range = max(detection_ranges) if detection_ranges else 0
        avg_range = np.mean([r for r in detection_ranges if r > 0]) if any(r > 0 for r in detection_ranges) else 0
        effective_fov = sum(1 for r in detection_ranges if r > camera.spec.max_range * 0.1)
        coverage_area = np.trapz([r for r in detection_ranges if r > 0], dx=2) if detection_ranges else 0
        
        # Add statistics text box
        stats_text = (f'Camera Specifications:\n'
                     f'• Sensor: {camera.spec.sensor_type.value}\n'
                     f'• Resolution: {camera.spec.resolution[0]}×{camera.spec.resolution[1]}\n'
                     f'• Max Range: {camera.spec.max_range:.0f}m\n'
                     f'• FOV: {camera.spec.field_of_view[0]:.1f}°×{camera.spec.field_of_view[1]:.1f}°\n\n'
                     f'Performance Metrics:\n'
                     f'• Max Detection: {max_range:.0f}m\n'
                     f'• Avg Range: {avg_range:.0f}m\n'
                     f'• Effective FOV: {effective_fov}°\n'
                     f'• Coverage Score: {coverage_area/1000:.1f}k')
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        ax.set_xlabel('Relative Angle (degrees)', fontsize=11)
        ax.set_ylabel('Detection Range (meters)', fontsize=11)
        ax.set_title(f'{camera.camera_id} - Detection Range Profile', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')

    def create_detection_range_summary(self, elevation_slice: float = 200.0, output_dir: str = 'images') -> str:
        """Create a summary comparison of all cameras' detection ranges"""
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        all_stats = []
        
        for i, camera in enumerate(self.cameras):
            if i < len(axes):
                angles = np.linspace(-90, 90, 181)
                detection_ranges = []
                
                for angle in angles:
                    test_range = camera.spec.max_range * 0.8
                    actual_angle = np.radians(angle + camera.orientation[0])
                    test_x = camera.position[0] + test_range * np.cos(actual_angle)
                    test_y = camera.position[1] + test_range * np.sin(actual_angle)
                    test_pos = np.array([test_x, test_y, elevation_slice])
                    
                    detection = self._check_detection_feasibility(camera, test_pos)
                    detection_ranges.append(detection['range'] if detection['detectable'] else 0)
                
                # Plot individual camera
                axes[i].plot(angles, detection_ranges, linewidth=2, 
                            color=plt.cm.tab10(i), label=f'Camera {i+1}')
                axes[i].fill_between(angles, detection_ranges, alpha=0.3, color=plt.cm.tab10(i))
                
                # Calculate stats
                max_range = max(detection_ranges) if detection_ranges else 0
                avg_range = np.mean([r for r in detection_ranges if r > 0]) if any(r > 0 for r in detection_ranges) else 0
                effective_fov = sum(1 for r in detection_ranges if r > camera.spec.max_range * 0.1)
                
                all_stats.append({
                    'camera_id': i+1,
                    'max_range': max_range,
                    'avg_range': avg_range,
                    'effective_fov': effective_fov
                })
                
                axes[i].set_title(f'Camera {i+1}', fontsize=12, fontweight='bold')
                axes[i].set_xlabel('Angle (°)', fontsize=10)
                axes[i].set_ylabel('Range (m)', fontsize=10)
                axes[i].grid(True, alpha=0.3)
                axes[i].set_ylim(0, max([camera.spec.max_range for camera in self.cameras]) * 1.1)
        
        plt.suptitle('Detection Range Comparison - All Cameras', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save summary
        summary_file = os.path.join(output_dir, 'detection_range_summary_all_cameras.png')
        fig.savefig(summary_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        # Print summary statistics
        print("\nDETECTION RANGE SUMMARY:")
        print("=" * 50)
        for stats in all_stats:
            print(f"Camera {stats['camera_id']:2d}: Max={stats['max_range']:4.0f}m, "
                  f"Avg={stats['avg_range']:4.0f}m, FOV={stats['effective_fov']:3.0f}°")
        
        return summary_file

    def visualize_sensor_coverage_single_representative(self, elevation_slice: float = 200.0) -> plt.Figure:
        """
        Show one representative camera with clear explanation of what detection range means
        This is good for audiences who need to understand the concept clearly
        """
        
        fig = plt.figure(figsize=(18, 10))
        
        # Main layout: 2 rows, 3 columns
        # Top row: Array overview, Representative camera detail, Detection explanation
        # Bottom row: Detection range plot spanning 2 columns, Statistics
        
        # Array overview (top left)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_full_array_overview(ax1)
        
        # Representative camera detail (top middle)
        ax2 = plt.subplot(2, 3, 2)
        representative_camera = self.cameras[0]  # Use first camera as representative
        self._plot_camera_context(ax2, representative_camera, 0)
        
        # Detection explanation (top right)
        ax3 = plt.subplot(2, 3, 3)
        self._plot_detection_explanation(ax3)
        
        # Detection range plot (bottom, spanning 2 columns)
        ax4 = plt.subplot(2, 3, (4, 5))
        self._plot_single_camera_detection_range(ax4, representative_camera, elevation_slice)
        
        # Performance summary (bottom right)
        ax5 = plt.subplot(2, 3, 6)
        self._plot_array_performance_summary(ax5, elevation_slice)
        
        plt.tight_layout(pad=2.0)
        return fig

    def _plot_full_array_overview(self, ax):
        """Plot overview of entire sensor array"""
        # Similar to existing sensor layout plot but more compact
        all_x = [cam.position[0] for cam in self.cameras] + [self.bounds.x_min, self.bounds.x_max]
        all_y = [cam.position[1] for cam in self.cameras] + [self.bounds.y_min, self.bounds.y_max]
        
        x_margin = (max(all_x) - min(all_x)) * 0.1
        y_margin = (max(all_y) - min(all_y)) * 0.1
        
        ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
        ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
        
        # Plot all cameras
        for i, camera in enumerate(self.cameras):
            color = 'red' if i == 0 else 'blue'
            size = 12 if i == 0 else 8
            ax.plot(camera.position[0], camera.position[1], 'o', 
                   color=color, markersize=size, markeredgecolor='black')
            ax.annotate(f'{i+1}', (camera.position[0], camera.position[1]),
                       xytext=(0, 0), textcoords='offset points',
                       fontsize=8, ha='center', va='center', color='white', fontweight='bold')
        
        # Surveillance area
        rect = plt.Rectangle((self.bounds.x_min, self.bounds.y_min), 
                            self.bounds.x_max - self.bounds.x_min,
                            self.bounds.y_max - self.bounds.y_min,
                            fill=False, edgecolor='green', linewidth=2)
        ax.add_patch(rect)
        
        ax.set_title('Complete Sensor Array\n(Camera 1 highlighted)', fontsize=12, fontweight='bold')
        ax.set_xlabel('X (meters)', fontsize=10)
        ax.set_ylabel('Y (meters)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    def _plot_detection_explanation(self, ax):
        """Plot explanation of what detection range means"""
        ax.axis('off')
        
        explanation_text = """
DETECTION RANGE EXPLAINED

The detection range plot shows how far 
each camera can detect a drone target 
at different angles.

KEY FACTORS:
• Field of View Limits
  Only within camera's FOV cone
  
• Distance Attenuation  
  Atmospheric effects reduce range
  
• Target Size Requirements
  Minimum apparent size needed
  
• Environmental Conditions
  Weather, lighting, humidity
  
• Sensor Limitations
  Resolution, noise thresholds

INTERPRETING THE PLOT:
• Peak = Optimal detection conditions
• Zero = Outside detection capability  
• Variations = Real-world limitations

This profile represents ALL cameras
in the array (similar specifications).
        """
        
        ax.text(0.05, 0.95, explanation_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

    def _plot_array_performance_summary(self, ax, elevation_slice):
        """Plot summary statistics for the entire array"""
        
        # Calculate coverage statistics
        total_cameras = len(self.cameras)
        total_max_range = sum(cam.spec.max_range for cam in self.cameras)
        avg_max_range = total_max_range / total_cameras if total_cameras > 0 else 0
        
        # Test coverage at center point
        center_point = np.array([(self.bounds.x_min + self.bounds.x_max)/2,
                                (self.bounds.y_min + self.bounds.y_max)/2,
                                elevation_slice])
        
        cameras_covering_center = 0
        for camera in self.cameras:
            detection = self._check_detection_feasibility(camera, center_point)
            if detection['detectable']:
                cameras_covering_center += 1
        
        redundancy = cameras_covering_center / total_cameras if total_cameras > 0 else 0
        
        # Create bar chart of key metrics
        metrics = ['Total\nCameras', 'Avg Max\nRange (km)', 'Center Point\nRedundancy', 'Array\nCoverage']
        values = [total_cameras, avg_max_range/1000, redundancy, 0.85]  # 0.85 is estimated coverage
        colors = ['blue', 'green', 'orange', 'red']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            if metrics[bars.index(bar)] == 'Total\nCameras':
                label = f'{int(value)}'
            elif 'Range' in metrics[bars.index(bar)]:
                label = f'{value:.1f}'
            else:
                label = f'{value:.2f}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                    label, ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('Array Performance Summary', fontsize=12, fontweight='bold')
        ax.set_ylabel('Normalized Values', fontsize=10)


def create_standard_sensor_array(simulation_bounds, array_type: str = "perimeter") -> VirtualSensorArray:
    """Create a standard sensor array configuration"""
    
    sensor_array = VirtualSensorArray(simulation_bounds)
    
    if array_type == "perimeter":
        sensor_array.setup_perimeter_array(num_cameras=8, 
                                         sensor_type=SensorType.VISIBLE_SPECTRUM)
    elif array_type == "triangulation":
        sensor_array.setup_triangulation_array(baseline_distance=300.0,
                                             sensor_type=SensorType.VISIBLE_SPECTRUM)
    elif array_type == "mixed":
        # Mixed sensor types
        sensor_array.setup_perimeter_array(num_cameras=6, 
                                         sensor_type=SensorType.VISIBLE_SPECTRUM)
        # Add thermal cameras at key positions
        thermal_array = VirtualSensorArray(simulation_bounds)
        thermal_array.setup_triangulation_array(baseline_distance=250.0,
                                               sensor_type=SensorType.THERMAL_INFRARED)
        # Merge thermal cameras
        for camera in thermal_array.cameras[:4]:  # Take first 4 thermal cameras
            sensor_array.cameras.append(camera)
    
    return sensor_array

# Example usage and testing
if __name__ == "__main__":
    from drone_trajectory_generator import SimulationBounds, DroneSwarmGenerator, FlightPattern
    
    # Create simulation environment
    bounds = SimulationBounds()
    
    # Test sensor array creation
    print("Creating virtual sensor array...")
    sensor_array = create_standard_sensor_array(bounds, "perimeter")
    
    print(f"Created sensor array with {len(sensor_array.cameras)} cameras")
    for cam in sensor_array.cameras:
        print(f"  {cam.camera_id}: {cam.spec.sensor_type.value} at {cam.position}")
    
    # Generate some test drone trajectories
    drone_gen = DroneSwarmGenerator(bounds)
    trajectory_data = drone_gen.generate_swarm_trajectories(
        num_drones=15,
        pattern=FlightPattern.COORDINATED_ATTACK,
        duration=10.0,
        timestep=0.5,
        drone_type='small'
    )
    
    # Test sensor observations
    print("\nTesting sensor observations...")
    trajectories = trajectory_data['trajectories']
    times = trajectory_data['times']
    
    # Sample a few time points
    test_times = [0, len(times)//4, len(times)//2, 3*len(times)//4, len(times)-1]
    
    for t_idx in test_times:
        drone_positions = trajectories[:, t_idx, :]
        timestamp = times[t_idx]
        
        observations = sensor_array.observe_targets(drone_positions, timestamp)
        stats = sensor_array.get_detection_statistics(observations)
        
        print(f"Time {timestamp:5.1f}s: {stats['total_detections']:2d} detections "
              f"from {stats['active_cameras']:2d}/{stats['total_cameras']:2d} cameras, "
              f"avg confidence: {stats['average_confidence']:.2f}")
    
    # Create visualization
    print("\nGenerating sensor coverage visualization...")
    try:
        fig = sensor_array.visualize_sensor_coverage()
        fig.savefig('sensor_coverage_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("Sensor coverage visualization saved as 'sensor_coverage_analysis.png'")
    except Exception as e:
        print(f"Visualization error: {e}")
    
    print("Sensor simulation system ready!")