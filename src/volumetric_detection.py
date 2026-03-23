"""
Volumetric Motion Detection Pipeline for Multi-Drone Swarm Engagement

Implements:
- Probabilistic occupancy grids with Bayesian updates
- Space carving algorithms for 3D reconstruction
- Multi-view geometry for target triangulation
- Real-time processing optimized for desktop simulation

Author: Research implementation for academic publication
Performance target: 30-50 simultaneous drones on desktop hardware
"""

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from multiprocessing import cpu_count
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
from enum import Enum
import time
from collections import defaultdict, deque
import scipy.spatial
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN

# Import the modularized classes
from bayesian_occupancy_updater import BayesianOccupancyUpdater, VoxelGrid
from space_carving import SpaceCarvingAlgorithm
from multi_view_triangulator import MultiViewTriangulator
from temporal_target_tracker import TemporalTargetTracker
from performance_monitor import PerformanceMonitor, PerformanceMetrics

class DetectionMethod(Enum):
    OCCUPANCY_GRID = "occupancy_grid"
    SPACE_CARVING = "space_carving"
    TRIANGULATION = "triangulation"
    HYBRID = "hybrid"

@dataclass
class DetectedTarget:
    """3D target detection result"""
    position: np.ndarray  # [x, y, z] world coordinates
    velocity: np.ndarray  # [vx, vy, vz] estimated velocity
    confidence: float     # Overall detection confidence [0-1]
    volume_estimate: float  # Estimated target volume (m³)
    contributing_sensors: List[str]  # Sensor IDs that detected this target
    detection_method: DetectionMethod  # Primary detection method used
    covariance_matrix: np.ndarray  # Position uncertainty covariance
    timestamp: float      # Detection timestamp
    target_id: Optional[int] = None  # Tracking ID if available

class VolumetricDetectionPipeline:
    """
    Main volumetric detection pipeline implementing multiple detection algorithms
    Optimized for real-time processing on desktop hardware
    """
    
    def __init__(self, simulation_bounds, voxel_resolution: float = 2.0,
                 temporal_decay_rate: float = 0.95, detection_threshold: float = 0.7):
        self.bounds = simulation_bounds
        self.voxel_resolution = voxel_resolution
        self.temporal_decay_rate = temporal_decay_rate
        self.detection_threshold = detection_threshold
        
        # Initialize 3D voxel grid
        self.grid = self._initialize_voxel_grid()
        
        # Detection algorithms
        self.occupancy_updater = BayesianOccupancyUpdater(self.grid)
        self.space_carver = SpaceCarvingAlgorithm(self.grid)
        self.triangulator = MultiViewTriangulator(simulation_bounds)
        
        # Target tracking
        self.target_tracker = TemporalTargetTracker()
        self.detection_history = deque(maxlen=100)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Algorithm parameters
        self.bayesian_params = {
            'prob_occupied_given_detection': 0.9,
            'prob_occupied_given_no_detection': 0.02,
            'prior_probability': 0.1,
            'minimum_evidence': 0.05
        }
        
        # Lower detection threshold
        self.detection_threshold = 0.55
        
        self.space_carving_params = {
            'carving_threshold': 0.3,
            'consistency_threshold': 0.6,
            'erosion_rate': 0.1
        }

    def process_sensor_observations_parallel(self, observations, timestamp, method, max_workers=None):
        """
        Parallel version of process_sensor_observations for better CPU utilization

        Args:
            observations: List of sensor observations
            timestamp: Current timestamp
            method: Detection method to use
            max_workers: Maximum worker threads (default: auto-detect)

        Returns:
            List of detected targets (same format as original)
        """
        if max_workers is None:
            max_workers = min(cpu_count(), 8)

        # Set optimal threading for numerical operations
        original_threads = os.environ.get('OMP_NUM_THREADS')
        os.environ['OMP_NUM_THREADS'] = str(max_workers)

        try:
            if method == DetectionMethod.TRIANGULATION:
                return self._parallel_triangulation(observations, timestamp, max_workers)

            elif method == DetectionMethod.SPACE_CARVING:
                return self._parallel_space_carving(observations, timestamp, max_workers)

            elif method == DetectionMethod.HYBRID:
                return self._parallel_hybrid(observations, timestamp, max_workers)

            else:
                # Fallback to original method
                return self.process_sensor_observations(observations, timestamp, method)

        finally:
            # Restore original threading setting
            if original_threads:
                os.environ['OMP_NUM_THREADS'] = original_threads
            elif 'OMP_NUM_THREADS' in os.environ:
                del os.environ['OMP_NUM_THREADS']

    def _parallel_space_carving(self, observations, timestamp, max_workers):
        """SIMPLIFIED space carving - directly mark areas around detections"""
        
        # Step 1: Get all detection positions
        detection_positions = []
        for obs in observations:
            for obj in obs.detected_objects:
                detection_positions.append(obj['world_position'])
        
        if not detection_positions:
            return []
        
        detection_positions = np.array(detection_positions)
        print(f"    Processing {len(detection_positions)} detection positions for space carving")
        
        # Step 2: SIMPLIFIED APPROACH - Just create targets directly around each detection
        carved_targets = []
        cluster_radius = self.voxel_resolution * 2  # Smaller clustering radius
        
        # Group nearby detections
        used_detections = set()
        
        for i, det_pos in enumerate(detection_positions):
            if i in used_detections:
                continue
                
            # Find all detections within cluster radius
            cluster_positions = [det_pos]
            used_detections.add(i)
            
            for j, other_pos in enumerate(detection_positions):
                if j != i and j not in used_detections:
                    distance = np.linalg.norm(det_pos - other_pos)
                    if distance < cluster_radius:
                        cluster_positions.append(other_pos)
                        used_detections.add(j)
            
            # Create a target for this cluster
            if len(cluster_positions) >= 1:  # Very lenient requirement
                cluster_center = np.mean(cluster_positions, axis=0)
                
                # Calculate confidence based on number of supporting observations
                supporting_cameras = 0
                total_confidence = 0
                
                for obs in observations:
                    for obj in obs.detected_objects:
                        dist_to_cluster = np.linalg.norm(np.array(obj['world_position']) - cluster_center)
                        if dist_to_cluster < self.voxel_resolution * 3:
                            supporting_cameras += 1
                            # Get confidence from this observation
                            if len(obs.confidence_scores) > 0:
                                total_confidence += np.mean(obs.confidence_scores)
                
                if supporting_cameras > 0:
                    avg_confidence = total_confidence / supporting_cameras
                    # Boost confidence for multiple supporting cameras
                    final_confidence = min(0.8, avg_confidence * (1 + 0.1 * supporting_cameras))
                    
                    class SimpleDetection:
                        def __init__(self, position, confidence):
                            self.position = position
                            self.confidence = final_confidence
                    
                    carved_targets.append(SimpleDetection(cluster_center, final_confidence))
        
        print(f"    Space carving created {len(carved_targets)} targets from {len(detection_positions)} detections")
        
        return carved_targets

    def _get_focused_voxel_coordinates(self, detection_positions, search_radius):
        """Get voxel coordinates only around detection areas - MUCH FASTER"""
        focused_coords = []
        # For each detection, only generate voxels in its neighborhood
        for det_pos in detection_positions:
            # Calculate voxel bounds around this detection
            min_coords = det_pos - search_radius
            max_coords = det_pos + search_radius

            # Clip to simulation bounds
            min_coords = np.maximum(min_coords, [self.bounds.x_min, self.bounds.y_min, self.bounds.z_min])
            max_coords = np.minimum(max_coords, [self.bounds.x_max, self.bounds.y_max, self.bounds.z_max])

            # Generate voxel coordinates in this region
            x_coords = np.arange(min_coords[0], max_coords[0], self.voxel_resolution)
            y_coords = np.arange(min_coords[1], max_coords[1], self.voxel_resolution)
            z_coords = np.arange(min_coords[2], max_coords[2], self.voxel_resolution)

            # Create local meshgrid
            if len(x_coords) > 0 and len(y_coords) > 0 and len(z_coords) > 0:
                X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
                local_coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
                focused_coords.extend(local_coords)

        # Remove duplicates
        if focused_coords:
            focused_coords = np.array(focused_coords)
            # Simple duplicate removal by rounding to voxel precision
            rounded = np.round(focused_coords / self.voxel_resolution) * self.voxel_resolution
            unique_coords = np.unique(rounded, axis=0)
            return unique_coords

        return np.array([])

    def _is_voxel_consistent_fast(self, voxel_coord, observations, detection_positions):
        """Fast voxel consistency check - OPTIMIZED and MORE LENIENT"""
        # Pre-filter: must be close to at least one detection
        distances_to_detections = np.linalg.norm(detection_positions - voxel_coord, axis=1)
        if np.min(distances_to_detections) > self.voxel_resolution * 5:  # Increased tolerance
            return False
        
        # More lenient consistency check
        consistent_count = 0
        for obs in observations:
            if len(obs.detected_objects) > 0:
                # Vectorized distance to all detections from this observer
                obs_positions = np.array([det['world_position'] for det in obs.detected_objects])
                distances = np.linalg.norm(obs_positions - voxel_coord, axis=1)
                
                if np.min(distances) < self.voxel_resolution * 4:  # More lenient
                    consistent_count += 1
                    if consistent_count >= 1:  # Only need 1 camera for initial detection
                        return True
        
        return consistent_count >= 1  # More lenient requirement

    def _calculate_voxel_confidence_fast(self, voxel_coord, observations):
        """Fast confidence calculation"""
        total_confidence = 0
        count = 0
        
        for obs in observations:
            if len(obs.confidence_scores) > 0:
                # Weight confidence by proximity to detections
                if len(obs.detected_objects) > 0:
                    obs_positions = np.array([det['world_position'] for det in obs.detected_objects])
                    distances = np.linalg.norm(obs_positions - voxel_coord, axis=1)
                    min_dist = np.min(distances)
                    
                    # Closer detections get higher weight
                    proximity_weight = max(0.1, 1.0 - min_dist / (self.voxel_resolution * 5))
                    weighted_confidence = np.mean(obs.confidence_scores) * proximity_weight
                    total_confidence += weighted_confidence
                    count += 1
        
        return total_confidence / count if count > 0 else 0.5

    def _parallel_triangulation(self, observations, timestamp, max_workers):
        """Parallelize triangulation processing"""
        
        # Create all possible observation pairs
        obs_pairs = []
        for i in range(len(observations)):
            for j in range(i + 1, len(observations)):
                if (len(observations[i].detected_objects) > 0 and 
                    len(observations[j].detected_objects) > 0):
                    obs_pairs.append((observations[i], observations[j]))
        
        if not obs_pairs:
            return []
        
        def triangulate_pair(pair):
            """Triangulate detections from a pair of observations"""
            obs1, obs2 = pair
            pair_results = []
            
            for det1 in obs1.detected_objects:
                for det2 in obs2.detected_objects:
                    # Simple triangulation calculation
                    pos1 = det1['world_position']
                    pos2 = det2['world_position']
                    estimated_pos = (pos1 + pos2) / 2
                    
                    # Calculate confidence based on detection quality
                    conf1 = obs1.confidence_scores[0] if len(obs1.confidence_scores) > 0 else 0.5
                    conf2 = obs2.confidence_scores[0] if len(obs2.confidence_scores) > 0 else 0.5
                    confidence = (conf1 + conf2) / 2
                    
                    pair_results.append({
                        'position': estimated_pos,
                        'confidence': confidence,
                        'method': 'triangulation'
                    })
            
            return pair_results
        
        # Process pairs in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(triangulate_pair, pair) for pair in obs_pairs]
            all_results = []
            for future in futures:
                all_results.extend(future.result())
        
        # Cluster and merge nearby detections
        return self._cluster_and_merge_detections(all_results)

    def _parallel_hybrid(self, observations, timestamp, max_workers):
        """Run triangulation and space carving in parallel, then fuse results"""
        
        def run_triangulation():
            return self._parallel_triangulation(observations, timestamp, max_workers // 2)
        
        def run_space_carving():
            return self._parallel_space_carving(observations, timestamp, max_workers // 2)
        
        # Run both methods simultaneously
        with ThreadPoolExecutor(max_workers=2) as executor:
            tri_future = executor.submit(run_triangulation)
            sc_future = executor.submit(run_space_carving)
            
            triangulation_results = tri_future.result()
            space_carving_results = sc_future.result()
        
        # Fuse the results from both methods - FIXED CALL
        return self._fuse_parallel_results(triangulation_results, space_carving_results)

    def _fuse_parallel_results(self, tri_results, sc_results):
        """Fuse triangulation and space carving results - SIMPLIFIED VERSION"""
        # Simple fusion - combine and remove duplicates
        all_results = list(tri_results) + list(sc_results)
        
        if not all_results:
            return []
        
        # Remove duplicates by clustering nearby detections
        return self._cluster_and_merge_detections([
            {'position': det.position, 'confidence': det.confidence} 
            for det in all_results
        ])

    def _get_all_voxel_coordinates(self):
        """Get all voxel coordinates in the detection volume - OPTIMIZED VERSION"""
        # Use numpy meshgrid for vectorized coordinate generation
        x_coords = np.arange(self.bounds.x_min, self.bounds.x_max, self.voxel_resolution)
        y_coords = np.arange(self.bounds.y_min, self.bounds.y_max, self.voxel_resolution)
        z_coords = np.arange(self.bounds.z_min, self.bounds.z_max, self.voxel_resolution)
        
        # Create 3D meshgrid - much faster than Python loops
        X, Y, Z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
        
        # Flatten and stack into coordinate array
        coords = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
        
        return coords

    def _is_voxel_consistent(self, voxel_coord, observations):
        """Check if voxel is consistent with observations - VECTORIZED VERSION"""
        # voxel_coord is now a numpy array [x, y, z] instead of individual arrays
        consistent_count = 0
        for obs in observations:
            if len(obs.detected_objects) > 0:
                # Vectorized distance calculation
                detection_positions = np.array([det['world_position'] for det in obs.detected_objects])
                
                # Calculate distances to all detections at once
                distances = np.linalg.norm(detection_positions - voxel_coord, axis=1)
                min_dist = np.min(distances)
                
                if min_dist < self.voxel_resolution * 2:  # Within 2 voxel sizes
                    consistent_count += 1
        
        return consistent_count >= 2  # At least 2 cameras see something nearby

    def _calculate_voxel_confidence(self, voxel_coord, observations):
        """Calculate confidence for a voxel (simplified)"""
        # Simplified confidence calculation
        total_confidence = 0
        count = 0
        
        for obs in observations:
            if len(obs.confidence_scores) > 0:
                total_confidence += np.mean(obs.confidence_scores)
                count += 1
        
        return total_confidence / count if count > 0 else 0.5

    def _convert_voxel_results_to_detections(self, voxel_results, threshold=0.5):
        """Convert voxel results to detection format"""
        # Filter by confidence and convert to your detection format
        class SimpleDetection:
            def __init__(self, position, confidence):
                self.position = position
                self.confidence = confidence
        
        detections = []
        for result in voxel_results:
            if result['confidence'] >= threshold:
                detections.append(SimpleDetection(result['position'], result['confidence']))
        
        return detections

    def _cluster_and_merge_detections(self, results):
        """Cluster nearby detections and merge them"""
        # Simplified clustering - group detections within voxel_resolution distance
        if not results:
            return []
        
        clusters = []
        used = set()
        
        for i, result in enumerate(results):
            if i in used:
                continue
                
            cluster = [result]
            used.add(i)
            
            for j, other in enumerate(results[i+1:], i+1):
                if j in used:
                    continue
                
                dist = np.linalg.norm(result['position'] - other['position'])
                if dist < self.voxel_resolution * 1.5:
                    cluster.append(other)
                    used.add(j)
            
            # Merge cluster into single detection
            if len(cluster) > 0:
                avg_pos = np.mean([c['position'] for c in cluster], axis=0)
                avg_conf = np.mean([c['confidence'] for c in cluster])
                
                class SimpleDetection:
                    def __init__(self, position, confidence):
                        self.position = position
                        self.confidence = confidence
                
                clusters.append(SimpleDetection(avg_pos, avg_conf))
        
        return clusters

    def _initialize_voxel_grid(self) -> VoxelGrid:
        """Initialize 3D voxel grid covering simulation volume"""
        x_range = self.bounds.x_max - self.bounds.x_min
        y_range = self.bounds.y_max - self.bounds.y_min
        z_range = self.bounds.z_max - self.bounds.z_min

        nx = int(np.ceil(x_range / self.voxel_resolution))
        ny = int(np.ceil(y_range / self.voxel_resolution))
        nz = int(np.ceil(z_range / self.voxel_resolution))

        origin = np.array([self.bounds.x_min, self.bounds.y_min, self.bounds.z_min])

        return VoxelGrid(
            origin=origin,
            resolution=self.voxel_resolution,
            dimensions=(nx, ny, nz)
        )
    
    def process_sensor_observations(self, observations: List, timestamp: float,
                                  method: DetectionMethod = DetectionMethod.HYBRID) -> List[DetectedTarget]:
        """Main processing pipeline for sensor observations"""
        start_time = time.time()
        
        # Apply temporal decay to existing occupancy probabilities
        self._apply_temporal_decay(timestamp)
        
        # Process observations based on selected method
        if method == DetectionMethod.OCCUPANCY_GRID:
            targets = self._process_occupancy_grid_method(observations, timestamp)
        elif method == DetectionMethod.SPACE_CARVING:
            targets = self._process_space_carving_method(observations, timestamp)
        elif method == DetectionMethod.TRIANGULATION:
            targets = self._process_triangulation_method(observations, timestamp)
        elif method == DetectionMethod.HYBRID:
            targets = self._process_hybrid_method(observations, timestamp)
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        # Update target tracking
        tracked_targets = self.target_tracker.update_tracks(targets, timestamp)
        
        # Record performance metrics
        processing_time = time.time() - start_time
        self.performance_monitor.record_frame(processing_time, len(targets), 
                                            len(observations), self.grid)
        
        # Store detection history
        self.detection_history.append({
            'timestamp': timestamp,
            'targets': tracked_targets,
            'method': method,
            'processing_time': processing_time
        })
        
        return tracked_targets
    
    def _apply_temporal_decay(self, current_time: float):
        """Apply exponential temporal decay to occupancy probabilities"""
        time_diff = current_time - self.grid.last_update_time
        decay_mask = time_diff > 0.1
        decay_factor = np.power(self.temporal_decay_rate, time_diff)
        prior = self.bayesian_params['prior_probability']
        self.grid.occupancy_probs[decay_mask] = (
            prior + (self.grid.occupancy_probs[decay_mask] - prior) * decay_factor[decay_mask])
        self.grid.confidence_scores *= np.power(0.98, time_diff)
    
    def _process_occupancy_grid_method(self, observations: List, timestamp: float) -> List[DetectedTarget]:
        """Process using Bayesian occupancy grid updates"""
        self.occupancy_updater.update_grid(observations, timestamp)
        targets = self._extract_targets_from_occupancy_grid(timestamp)
        return targets
    
    def _extract_targets_from_occupancy_grid(self, timestamp: float) -> List[DetectedTarget]:
        """Extract target detections from high-probability grid regions"""
        high_prob_mask = self.grid.occupancy_probs > self.detection_threshold
        high_conf_mask = self.grid.confidence_scores > 0.3
        detection_mask = high_prob_mask | (self.grid.occupancy_probs > 0.4)
        
        if not np.any(detection_mask):
            return []
        
        detected_coords = np.where(detection_mask)
        detected_positions = self._voxel_to_world_coordinates(detected_coords)
        targets = self._cluster_detections_enhanced(detected_positions, detection_mask, timestamp)
        return targets
    
    def _cluster_detections_enhanced(self, positions: np.ndarray, mask: np.ndarray, 
                                   timestamp: float) -> List[DetectedTarget]:
        """Enhanced clustering with better noise filtering"""
        if len(positions) == 0:
            return []
        
        targets = []
        clustering_distance = self.voxel_resolution * 3
        
        try:
            clustering = DBSCAN(eps=clustering_distance, min_samples=2).fit(positions)
            labels = clustering.labels_
            
            for label in set(labels):
                if label == -1:
                    continue
                
                cluster_mask = labels == label
                cluster_positions = positions[cluster_mask]
                
                if len(cluster_positions) < 2:
                    continue
                
                centroid = np.mean(cluster_positions, axis=0)
                volume_estimate = len(cluster_positions) * (self.voxel_resolution ** 3)
                
                voxel_coord = self._world_to_voxel_coordinates(centroid.reshape(1, -1))[0]
                cluster_prob = self.grid.occupancy_probs[
                    voxel_coord[0], voxel_coord[1], voxel_coord[2]]
                confidence = min(0.9, 0.2 + cluster_prob * 0.7)
                
                velocity = self._estimate_velocity_from_history(centroid, timestamp)
                position_std = np.std(cluster_positions, axis=0)
                position_std = np.maximum(position_std, self.voxel_resolution * 0.5)
                covariance = np.diag(position_std ** 2)
                
                target = DetectedTarget(
                    position=centroid,
                    velocity=velocity,
                    confidence=confidence,
                    volume_estimate=volume_estimate,
                    contributing_sensors=[],
                    detection_method=DetectionMethod.OCCUPANCY_GRID,
                    covariance_matrix=covariance,
                    timestamp=timestamp
                )
                targets.append(target)
                
        except ImportError:
            targets = self._cluster_detections_simple(positions, mask, timestamp)
        
        return targets
    
    def _cluster_detections_simple(self, positions: np.ndarray, mask: np.ndarray, 
                                 timestamp: float) -> List[DetectedTarget]:
        """Simple clustering fallback"""
        from scipy.cluster.hierarchy import fcluster, linkage
        
        if len(positions) < 2:
            return []
        
        linkage_matrix = linkage(positions, method='ward')
        cluster_labels = fcluster(linkage_matrix, self.voxel_resolution * 4, criterion='distance')
        
        targets = []
        for cluster_id in set(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_positions = positions[cluster_mask]
            
            if len(cluster_positions) < 3:
                continue
            
            centroid = np.mean(cluster_positions, axis=0)
            volume_estimate = len(cluster_positions) * (self.voxel_resolution ** 3)
            
            target = DetectedTarget(
                position=centroid,
                velocity=np.zeros(3),
                confidence=0.5,
                volume_estimate=volume_estimate,
                contributing_sensors=[],
                detection_method=DetectionMethod.OCCUPANCY_GRID,
                covariance_matrix=np.eye(3) * (self.voxel_resolution ** 2),
                timestamp=timestamp
            )
            targets.append(target)
        
        return targets
    
    def _process_space_carving_method(self, observations: List, timestamp: float) -> List[DetectedTarget]:
        """Process using space carving algorithm"""
        carved_volume = self.space_carver.carve_space(observations, timestamp)
        targets = self._extract_targets_from_carved_space(carved_volume, timestamp)
        return targets
    
    def _process_triangulation_method(self, observations: List, timestamp: float) -> List[DetectedTarget]:
        """Process using multi-view triangulation"""
        object_groups = self._group_observations_by_object(observations)
        targets = []
        for group in object_groups:
            if len(group) >= 2:
                target = self.triangulator.triangulate_target(group, timestamp)
                if target is not None:
                    targets.append(target)
        return targets
    
    def _process_hybrid_method(self, observations: List, timestamp: float) -> List[DetectedTarget]:
        """Process using hybrid approach combining multiple methods"""
        triangulation_targets = self._process_triangulation_method(observations, timestamp)
        self.occupancy_updater.update_grid(observations, timestamp)
        carved_volume = self.space_carver.carve_space(observations, timestamp)
        hybrid_targets = self._fuse_detection_results(triangulation_targets, carved_volume, timestamp)
        return hybrid_targets
    
    def _group_observations_by_object(self, observations: List) -> List[List]:
        """Group observations by detected objects for triangulation"""
        max_objects = max(len(obs.detected_objects) for obs in observations) if observations else 0
        object_groups = [[] for _ in range(max_objects)]
        
        for obs in observations:
            for i, obj in enumerate(obs.detected_objects):
                if i < len(object_groups):
                    object_groups[i].append(obs)
        
        return [group for group in object_groups if len(group) >= 2]
    
    def _fuse_detection_results(self, triangulation_targets: List[DetectedTarget], 
                               carved_volume: np.ndarray, timestamp: float) -> List[DetectedTarget]:
        """Fuse results from multiple detection methods"""
        fused_targets = triangulation_targets.copy()
        carving_targets = self._extract_targets_from_carved_space(carved_volume, timestamp)
        occupancy_targets = self._extract_targets_from_occupancy_grid(timestamp)
        
        additional_targets = carving_targets + occupancy_targets
        
        for add_target in additional_targets:
            is_duplicate = False
            for tri_target in triangulation_targets:
                distance = np.linalg.norm(add_target.position - tri_target.position)
                if distance < 15.0:
                    if add_target.detection_method == DetectionMethod.SPACE_CARVING:
                        tri_target.confidence = min(0.95, tri_target.confidence * 1.3)
                    is_duplicate = True
                    break
            
            if not is_duplicate and add_target.confidence > 0.1:
                if self._check_carving_support(add_target.position, carved_volume):
                    add_target.confidence *= 1.2
                fused_targets.append(add_target)
        
        return fused_targets
    
    def _check_carving_support(self, position: np.ndarray, carved_volume: np.ndarray) -> bool:
        """Check if a position is supported by carved volume"""
        voxel_coord = self._world_to_voxel_coordinates(position.reshape(1, -1))[0]
        
        if not (0 <= voxel_coord[0] < carved_volume.shape[0] and
                0 <= voxel_coord[1] < carved_volume.shape[1] and
                0 <= voxel_coord[2] < carved_volume.shape[2]):
            return False
        
        support_count = 0
        total_checked = 0
        
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                for dz in range(-1, 2):
                    x, y, z = voxel_coord[0] + dx, voxel_coord[1] + dy, voxel_coord[2] + dz
                    if (0 <= x < carved_volume.shape[0] and
                        0 <= y < carved_volume.shape[1] and
                        0 <= z < carved_volume.shape[2]):
                        if carved_volume[x, y, z]:
                            support_count += 1
                        total_checked += 1
        
        return (support_count / max(1, total_checked)) > 0.3
    
    def _extract_targets_from_carved_space(self, carved_volume: np.ndarray, 
                                          timestamp: float) -> List[DetectedTarget]:
        """Extract targets from space carving results"""
        carved_coords = np.where(carved_volume)
        
        if len(carved_coords[0]) == 0:
            return []
        
        world_positions = self._voxel_to_world_coordinates(carved_coords)
        mask = np.ones(len(world_positions), dtype=bool)
        targets = self._cluster_detections_simple(world_positions, mask, timestamp)
        
        for target in targets:
            target.detection_method = DetectionMethod.SPACE_CARVING
        
        return targets
    
    def _voxel_to_world_coordinates(self, voxel_coords: Tuple) -> np.ndarray:
        """Convert voxel grid coordinates to world coordinates"""
        voxel_array = np.column_stack(voxel_coords)
        world_coords = (self.grid.origin + 
                       voxel_array * self.voxel_resolution + 
                       self.voxel_resolution / 2)
        return world_coords
    
    def _world_to_voxel_coordinates(self, world_coords: np.ndarray) -> np.ndarray:
        """Convert world coordinates to voxel grid indices"""
        relative_coords = world_coords - self.grid.origin
        voxel_coords = (relative_coords / self.voxel_resolution).astype(int)
        
        voxel_coords[:, 0] = np.clip(voxel_coords[:, 0], 0, self.grid.dimensions[0] - 1)
        voxel_coords[:, 1] = np.clip(voxel_coords[:, 1], 0, self.grid.dimensions[1] - 1)
        voxel_coords[:, 2] = np.clip(voxel_coords[:, 2], 0, self.grid.dimensions[2] - 1)
        
        return voxel_coords
    
    def _estimate_velocity_from_history(self, position: np.ndarray, timestamp: float) -> np.ndarray:
        """Estimate velocity using recent detection history"""
        if len(self.detection_history) < 2:
            return np.zeros(3)
        
        min_distance = float('inf')
        best_match = None
        
        for hist_entry in reversed(list(self.detection_history)[-5:]):
            time_diff = timestamp - hist_entry['timestamp']
            if time_diff <= 0 or time_diff > 2.0:
                continue
                
            for target in hist_entry['targets']:
                distance = np.linalg.norm(target.position - position)
                if distance < min_distance and distance < 20.0:
                    min_distance = distance
                    best_match = (target.position, time_diff)
        
        if best_match is not None:
            old_position, time_diff = best_match
            velocity = (position - old_position) / time_diff
            return velocity
        
        return np.zeros(3)
    
    def debug_occupancy_grid(self):
        """Debug function to analyze occupancy grid state"""
        print("=== OCCUPANCY GRID DEBUG ===")
        print(f"Grid dimensions: {self.grid.dimensions}")
        print(f"Total voxels: {np.prod(self.grid.dimensions)}")
        
        probs = self.grid.occupancy_probs.flatten()
        print(f"Probability stats:")
        print(f"  Min: {np.min(probs):.3f}")
        print(f"  Max: {np.max(probs):.3f}")
        print(f"  Mean: {np.mean(probs):.3f}")
        
        high_prob = np.sum(self.grid.occupancy_probs > 0.6)
        threshold_voxels = np.sum(self.grid.occupancy_probs > self.detection_threshold)
        print(f"  Voxels > 0.6: {high_prob} ({high_prob/np.prod(self.grid.dimensions)*100:.3f}%)")
        print(f"  Voxels > threshold ({self.detection_threshold}): {threshold_voxels}")
