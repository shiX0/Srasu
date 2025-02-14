import torch
import cv2
import numpy as np
import open3d as o3d
from typing import Tuple, List, Dict, Optional
import threading
import time
from scipy.optimize import least_squares
import os

class DepthEstimator:
    def __init__(self):
        """Initialize MiDaS depth estimation model"""
        try:
            # Initialize flags
            self.running = True
            self.visualization_initialized = False

            # Check for AMD GPU and force ROCm if available
            if hasattr(torch, 'has_rocm') and torch.has_rocm:
                print("Using AMD GPU with ROCm")
                self.device = torch.device("cuda:0")  # ROCm uses CUDA device naming
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.fastest = True
            elif torch.cuda.is_available():
                print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
                self.device = torch.device("cuda:0")
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.fastest = True
            else:
                print("WARNING: No GPU acceleration available. Performance will be slow.")
                self.device = torch.device("cpu")
            
            # Initialize MiDaS model with smaller memory footprint
            self.model_type = "MiDaS_small"  # Use small model for better performance
            self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type)
            self.midas = self.midas.to(self.device)
            self.midas.eval()

            # Load transforms
            self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = self.midas_transforms.small_transform

            print(f"Depth estimation model loaded successfully on {self.device}")

            # Initialize camera parameters (KITTI)
            self.fx = 718.856
            self.fy = 718.856
            self.cx = 607.1928
            self.cy = 185.2157
            self.camera_matrix = np.array([
                [self.fx, 0, self.cx],
                [0, self.fy, self.cy],
                [0, 0, 1]
            ])
            self.dist_coeffs = np.zeros(5)  # Assuming no distortion for now

            # Initialize visualization variables
            self.visualization_initialized = True
            self.vis_width = 1280  # Increased resolution
            self.vis_height = 720
            self.last_update_time = time.time()
            self.update_interval = 1.0 / 30.0

            # Create visualization window
            cv2.namedWindow("3D Scene Understanding", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("3D Scene Understanding", self.vis_width, self.vis_height)

            print("Visualization initialized successfully")

            # Initialize additional parameters for improved 3D detection
            self.orientation_bins = 12  # Number of orientation bins
            
            # Initialize lane guidance parameters
            self.lane_color = (0, 255, 0)  # Green for lanes
            self.lane_thickness = 2
            self.lane_alpha = 0.3

            # Initialize 3D reconstruction parameters
            self.current_pose = np.eye(4)
            self.trajectory_points = []
            self.point_cloud = []
            self.point_colors = []
            self.camera_poses = []
            self.keyframes = []
            self.keyframe_poses = []
            self.keyframe_keypoints = []
            self.keyframe_descriptors = []
            self.feature_detector = cv2.SIFT_create(nfeatures=2000)
            self.feature_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

            # Initialize Open3D visualizer with better settings
            self.vis3d = o3d.visualization.Visualizer()
            self.vis3d.create_window("3D Reconstruction", width=1280, height=720)
            
            # Configure render options
            render_option = self.vis3d.get_render_option()
            render_option.point_size = 3.0
            render_option.background_color = np.array([0.1, 0.1, 0.1])
            render_option.light_on = True
            render_option.point_show_normal = False
            render_option.mesh_show_wireframe = True
            
            # Configure view control
            view_control = self.vis3d.get_view_control()
            view_control.set_zoom(0.3)
            view_control.set_front([0, 0, -1])
            view_control.set_up([0, -1, 0])

            # Track active geometries
            self.active_geometries = set()
            self.geometry_objects = {}
            
            # Add orientation tracking
            self.previous_orientations = {}  # Track orientations by object ID
            self.orientation_history = {}    # Keep history of orientations
            self.orientation_window = 5      # Number of frames for smoothing
            self.orientation_threshold = np.pi/6  # Threshold for orientation change
            
            print("3D reconstruction initialized successfully")

            # Initialize 3D model cache
            self.model_cache = {}
            self.model_scales = {
                'person': 1.7,  # Average height in meters
                'car': 4.5,    # Average length in meters
                'truck': 7.0,  # Average length in meters
                'bus': 12.0,   # Average length in meters
                'stop_sign': 0.9,  # Standard height in meters
                'traffic_light': 3.0  # Standard height in meters
            }
            
            # Load 3D models
            self.load_3d_models()
            
            print("3D models loaded successfully")

        except Exception as e:
            print(f"Error initializing: {str(e)}")
            raise

    def __del__(self):
        """Cleanup visualization"""
        try:
            self.running = False
            # Clear all geometries
            for geom_id in list(self.active_geometries):
                self.vis3d.remove_geometry(self.geometry_objects[geom_id], False)
            self.active_geometries.clear()
            self.geometry_objects.clear()
            
            # Destroy visualizer window
            self.vis3d.destroy_window()
            
            cv2.destroyAllWindows()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass

    def update_visualization(self, frame: np.ndarray, depth_map: np.ndarray, 
                           detected_objects: List[Dict] = None, lane_info: Dict = None):
        """Update visualization with enhanced 3D perspective"""
        if not self.visualization_initialized:
            return

        try:
            current_time = time.time()
            if current_time - self.last_update_time < self.update_interval:
                return
            self.last_update_time = current_time

            # Resize frame to match visualization dimensions
            frame = cv2.resize(frame, (self.vis_width, self.vis_height))
            
            # Create black background for 3D visualization
            vis_img = np.zeros((self.vis_height, self.vis_width, 3), dtype=np.uint8)
            
            # Add ground plane with perspective
            self.draw_grid_overlay(vis_img)
            
            # Resize depth map to match visualization dimensions
            if depth_map is not None:
                depth_map = cv2.resize(depth_map, (self.vis_width, self.vis_height))
                
                # Create ground plane visualization
                ground_overlay = self.create_ground_plane_visualization(depth_map)
                if ground_overlay is not None:
                    # Ensure ground overlay has same dimensions
                    ground_overlay = cv2.resize(ground_overlay, (self.vis_width, self.vis_height))
                    vis_img = cv2.addWeighted(vis_img, 0.7, ground_overlay, 0.3, 0)
            
            # Draw lane guidance if available
            if lane_info:
                self.draw_lane_guidance(vis_img, lane_info, depth_map)
            
            # Sort objects by depth for proper occlusion handling
            if detected_objects:
                objects_with_depth = []
                for obj in detected_objects:
                    if 'bbox' in obj:
                        # Scale bounding box coordinates
                        x1, y1, x2, y2 = obj['bbox']
                        x1 = int(x1 * self.vis_width / frame.shape[1])
                        x2 = int(x2 * self.vis_width / frame.shape[1])
                        y1 = int(y1 * self.vis_height / frame.shape[0])
                        y2 = int(y2 * self.vis_height / frame.shape[0])
                        obj['bbox'] = [x1, y1, x2, y2]
                        
                        center_y = int((y1 + y2) / 2)
                        if depth_map is not None:
                            depth_value = self.estimate_robust_depth(depth_map, int((x1 + x2) / 2), center_y)
                            objects_with_depth.append((obj, depth_value))
                
                # Sort objects by depth (furthest first)
                if objects_with_depth:
                    objects_with_depth.sort(key=lambda x: x[1], reverse=True)
                    
                    # Draw objects with proper depth ordering
                    for obj, depth_value in objects_with_depth:
                        x1, y1, x2, y2 = obj['bbox']
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        
                        # Draw 3D bounding boxes
                        orientation = self.estimate_orientation(obj, depth_map) if depth_map is not None else 0
                        dimensions = self.estimate_3d_dimensions(obj['type'], x2-x1, y2-y1, depth_value)
                        bbox3d = self.create_oriented_3d_bbox(obj['bbox'], depth_value, dimensions, orientation)
                        projected_points = self.project_3d_bbox(bbox3d)
                        
                        if not np.any(np.isnan(projected_points)):  # Check for invalid projections
                            self.draw_oriented_3d_bbox(vis_img, projected_points, obj, orientation)
                            self.draw_enhanced_object_info(vis_img, obj, depth_value, orientation, center_x, y1)
            
            # Show visualization
            cv2.imshow("3D Scene Understanding", vis_img)
            
        except Exception as e:
            print(f"Error updating visualization: {str(e)}")

    def project_3d_bbox(self, bbox3d: np.ndarray) -> np.ndarray:
        """Project 3D bounding box corners to image plane"""
        # Create projection matrix
        P = np.array([[self.fx, 0, self.cx, 0],
                     [0, self.fy, self.cy, 0],
                     [0, 0, 1, 0]])
        
        # Project points
        points_2d = []
        for point in bbox3d:
            point_homogeneous = np.append(point, 1)
            point_projected = P @ point_homogeneous
            point_2d = point_projected[:2] / point_projected[2]
            points_2d.append(point_2d)
            
        return np.array(points_2d)

    def draw_grid_overlay(self, img: np.ndarray):
        """Draw perspective grid for better depth perception"""
        height, width = img.shape[:2]
        horizon_y = int(height * 0.25)  # Moved horizon line up to show more ground
        
        # Fill the ground area with a dark green color for better visibility
        ground_mask = np.zeros((height, width, 3), dtype=np.uint8)
        ground_points = np.array([[0, height],
                                [width, height],
                                [width, horizon_y],
                                [0, horizon_y]], dtype=np.int32)
        cv2.fillPoly(ground_mask, [ground_points], (0, 40, 0))  # Dark green ground
        cv2.addWeighted(img, 1, ground_mask, 0.3, 0, img)
        
        # Draw horizon line
        cv2.line(img, (0, horizon_y), (width, horizon_y), (50, 50, 50), 2)

    def draw_lane_guidance(self, img: np.ndarray, lane_info: Dict, depth_map: np.ndarray):
        """Draw lane guidance with depth awareness"""
        if 'left_lane' in lane_info and 'right_lane' in lane_info:
            # Draw lane boundaries with enhanced visibility
            for lane in [lane_info['left_lane'], lane_info['right_lane']]:
                points = np.array(lane, dtype=np.int32)
                # Create depth-aware color gradient
                for i in range(len(points)-1):
                    depth = depth_map[points[i][1], points[i][0]]
                    alpha = max(0.4, 1 - depth/50)  # Increased minimum visibility
                    color = tuple([int(c * alpha) for c in (0, 255, 255)])  # Yellow color
                    cv2.line(img, tuple(points[i]), tuple(points[i+1]), 
                            color, 3)  # Increased line thickness
            
            # Draw driving corridor with enhanced visibility
            if 'corridor' in lane_info:
                corridor = np.array(lane_info['corridor'], dtype=np.int32)
                overlay = img.copy()
                cv2.fillPoly(overlay, [corridor], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
                
                # Add corridor boundary lines
                cv2.polylines(img, [corridor], True, (0, 255, 0), 2)

    def estimate_orientation(self, obj: Dict, depth_map: np.ndarray) -> float:
        """Estimate object orientation using improved depth and shape analysis with temporal smoothing"""
        try:
            x1, y1, x2, y2 = obj['bbox']
            width = x2 - x1
            height = y2 - y1
            
            # Generate object ID based on position and type (for tracking)
            obj_id = f"{obj['type']}_{int(x1)}_{int(y1)}"
            
            # Get depth gradient across object with improved robustness
            depth_roi = depth_map[max(0, y1):min(depth_map.shape[0], y2),
                                max(0, x1):min(depth_map.shape[1], x2)]
            
            if depth_roi.size == 0:
                return self.get_previous_orientation(obj_id, 0.0)
            
            # Calculate depth gradients with larger kernel for stability
            grad_x = cv2.Sobel(depth_roi, cv2.CV_64F, 1, 0, ksize=5)
            grad_y = cv2.Sobel(depth_roi, cv2.CV_64F, 0, 1, ksize=5)
            
            # Calculate magnitude and direction of gradient
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x)
            
            # Use weighted average of gradients based on magnitude
            weights = magnitude / (np.sum(magnitude) + 1e-6)
            mean_direction = np.sum(direction * weights)
            
            # Calculate confidence based on gradient strength
            gradient_strength = np.mean(magnitude)
            confidence = np.clip(gradient_strength / (np.max(magnitude) + 1e-6), 0.1, 1.0)
            
            # Get aspect ratio for additional cues
            aspect_ratio = width / height
            
            # Determine orientation based on object type and features
            if obj['type'].lower() in ['car', 'truck', 'bus']:
                # For vehicles, use more sophisticated orientation estimation
                orientation = self.estimate_vehicle_orientation(
                    depth_roi, aspect_ratio, mean_direction, grad_x, grad_y)
            else:
                # For other objects, use simpler orientation estimation
                if aspect_ratio > 1.5 or aspect_ratio < 0.67:
                    # For clearly elongated objects, use aspect ratio
                    orientation = 0 if aspect_ratio > 1 else np.pi/2
                else:
                    # For more square objects, use gradient direction if confident
                    if confidence > 0.3:
                        orientation = mean_direction
                    else:
                        orientation = 0  # Default to forward orientation
            
            # Apply temporal smoothing
            smoothed_orientation = self.smooth_orientation(obj_id, orientation, confidence)
            
            return smoothed_orientation
            
        except Exception as e:
            print(f"Error in orientation estimation: {str(e)}")
            return 0.0

    def smooth_orientation(self, obj_id: str, new_orientation: float, confidence: float) -> float:
        """Apply temporal smoothing to orientation estimates"""
        try:
            # Initialize history if needed
            if obj_id not in self.orientation_history:
                self.orientation_history[obj_id] = []
                self.previous_orientations[obj_id] = new_orientation
                
            history = self.orientation_history[obj_id]
            prev_orientation = self.previous_orientations[obj_id]
            
            # Handle wraparound cases (e.g., between 0 and 2π)
            if abs(new_orientation - prev_orientation) > np.pi:
                if new_orientation > prev_orientation:
                    new_orientation -= 2 * np.pi
                else:
                    new_orientation += 2 * np.pi
                    
            # Check if orientation change is too sudden
            if abs(new_orientation - prev_orientation) > self.orientation_threshold:
                # If sudden change, only partially accept it based on confidence
                new_orientation = prev_orientation + \
                    (new_orientation - prev_orientation) * confidence
                    
            # Add to history
            history.append((new_orientation, confidence))
            if len(history) > self.orientation_window:
                history.pop(0)
                
            # Compute weighted average over history
            total_weight = sum(conf for _, conf in history)
            if total_weight > 0:
                smoothed = sum(ori * conf for ori, conf in history) / total_weight
            else:
                smoothed = new_orientation
                
            # Normalize angle to [0, 2π)
            smoothed = smoothed % (2 * np.pi)
            
            # Update previous orientation
            self.previous_orientations[obj_id] = smoothed
            
            return smoothed
            
        except Exception as e:
            print(f"Error in orientation smoothing: {str(e)}")
            return new_orientation

    def get_previous_orientation(self, obj_id: str, default: float) -> float:
        """Get the previous orientation for an object"""
        return self.previous_orientations.get(obj_id, default)

    def estimate_vehicle_orientation(self, depth_roi: np.ndarray, aspect_ratio: float,
                                   mean_direction: float, grad_x: np.ndarray, 
                                   grad_y: np.ndarray) -> float:
        """Estimate orientation specifically for vehicles"""
        try:
            # Calculate histogram of gradients for more robust orientation
            hist_bins = 32
            angles = np.arctan2(grad_y, grad_x).flatten()
            magnitudes = np.sqrt(grad_x**2 + grad_y**2).flatten()
            hist, _ = np.histogram(angles, bins=hist_bins, range=(-np.pi, np.pi), 
                                 weights=magnitudes)
            
            # Find dominant orientations (could be multiple peaks)
            peaks = []
            for i in range(hist_bins):
                if hist[i] > np.mean(hist) + np.std(hist):
                    angle = -np.pi + (2 * np.pi * i) / hist_bins
                    peaks.append((hist[i], angle))
            
            if len(peaks) > 0:
                # Sort peaks by magnitude
                peaks.sort(reverse=True)
                dominant_angle = peaks[0][1]
                
                # Determine if side view or front/back view
                if aspect_ratio > 1.8:  # Likely side view
                    # Use dominant gradient direction
                    orientation = dominant_angle
                    # Snap to 0 or π if close
                    if abs(orientation) < self.orientation_threshold or \
                       abs(abs(orientation) - np.pi) < self.orientation_threshold:
                        orientation = 0 if abs(orientation) < np.pi/2 else np.pi
                else:  # Likely front/back view
                    # Use depth gradient to determine front/back
                    mean_grad_x = np.mean(grad_x)
                    orientation = 0 if mean_grad_x > 0 else np.pi
            else:
                # Fallback to aspect ratio based estimation
                orientation = 0 if aspect_ratio > 1.2 else np.pi
                
            return orientation
            
        except Exception as e:
            print(f"Error in vehicle orientation estimation: {str(e)}")
            return 0.0

    def estimate_robust_depth(self, depth_map: np.ndarray, center_x: int, center_y: int) -> float:
        """Estimate robust depth using improved statistical analysis"""
        window_size = 7  # Increased window size for better statistics
        x1 = max(0, center_x - window_size)
        x2 = min(depth_map.shape[1], center_x + window_size)
        y1 = max(0, center_y - window_size)
        y2 = min(depth_map.shape[0], center_y + window_size)
        
        depth_window = depth_map[y1:y2, x1:x2]
        
        if depth_window.size == 0:
            return 30.0  # Default depth if window is invalid
        
        # Remove outliers using IQR method
        q1 = np.percentile(depth_window, 25)
        q3 = np.percentile(depth_window, 75)
        iqr = q3 - q1
        valid_depths = depth_window[
            (depth_window >= q1 - 1.5*iqr) & 
            (depth_window <= q3 + 1.5*iqr)
        ]
        
        if valid_depths.size > 0:
            # Use trimmed mean for robust depth estimation
            depth = np.mean(valid_depths)
        else:
            depth = np.median(depth_window)  # Fallback to median
            
        return np.clip(depth, 1.0, 50.0)  # Limit depth range

    def create_oriented_3d_bbox(self, bbox2d: List[int], depth: float, 
                              dimensions: Tuple[float, float, float], 
                              orientation: float) -> np.ndarray:
        """Create oriented 3D bounding box with improved accuracy and perspective handling"""
        x1, y1, x2, y2 = bbox2d
        width, height, length = dimensions
        
        # Calculate 3D center point with improved perspective correction
        center_x = ((x1 + x2) / 2 - self.cx) * depth / self.fx
        center_y = ((y1 + y2) / 2 - self.cy) * depth / self.fy
        center_z = depth
        
        # Apply ground plane constraint for better placement
        # Assume objects are on the ground plane
        if y2 > self.vis_height * 0.5:  # Only apply to objects in lower half of image
            ground_y = (y2 - self.cy) * depth / self.fy  # Y coordinate of bottom edge
            center_y = ground_y - height/2  # Adjust center to sit on ground
        
        # Create rotation matrix with improved orientation handling
        cos_rot = np.cos(orientation)
        sin_rot = np.sin(orientation)
        R = np.array([
            [cos_rot, 0, sin_rot],
            [0, 1, 0],
            [-sin_rot, 0, cos_rot]
        ])
        
        # Create 3D box corners with improved perspective handling
        corners = np.array([
            [-width/2, -height/2, -length/2],  # Front bottom left
            [width/2, -height/2, -length/2],   # Front bottom right
            [width/2, height/2, -length/2],    # Front top right
            [-width/2, height/2, -length/2],   # Front top left
            [-width/2, -height/2, length/2],   # Back bottom left
            [width/2, -height/2, length/2],    # Back bottom right
            [width/2, height/2, length/2],     # Back top right
            [-width/2, height/2, length/2]     # Back top left
        ])
        
        # Apply rotation and translation
        corners = (R @ corners.T).T + np.array([center_x, center_y, center_z])
        
        # Apply depth-based size adjustment for distant objects
        depth_scale = np.clip(1.0 - depth/100, 0.8, 1.0)  # Less aggressive scaling
        corners = corners * depth_scale
        
        return corners

    def draw_oriented_3d_bbox(self, img: np.ndarray, points_2d: np.ndarray, 
                            obj: Dict, orientation: float):
        """Draw oriented 3D bounding box and place 3D model"""
        try:
            # Draw bounding box as before
            edges = [(0,1), (1,2), (2,3), (3,0),  # front face
                    (4,5), (5,6), (6,7), (7,4),  # back face
                    (0,4), (1,5), (2,6), (3,7)]  # connecting edges
            
            # Calculate face colors with depth-based shading
            base_color = np.array(obj['color'])
            front_color = tuple(map(int, base_color * 1.2))
            back_color = tuple(map(int, base_color * 0.6))
            
            # Draw edges with proper depth ordering
            for i, j in edges:
                pt1 = tuple(points_2d[i].astype(int))
                pt2 = tuple(points_2d[j].astype(int))
                cv2.line(img, pt1, pt2, front_color, 2)
            
            # Place 3D model if available
            if 'bbox' in obj and 'type' in obj:
                depth = self.estimate_robust_depth(
                    self.last_depth_map,
                    int((obj['bbox'][0] + obj['bbox'][2]) / 2),
                    int((obj['bbox'][1] + obj['bbox'][3]) / 2)
                )
                
                try:
                    model = self.place_3d_model(
                        obj['type'].lower(),
                        obj['bbox'],
                        depth,
                        orientation
                    )
                    
                    if model is not None:
                        # Generate unique ID for this object instance
                        geom_id = f"model_{id(obj)}"
                        
                        # Remove old geometry if it exists
                        if geom_id in self.active_geometries:
                            self.vis3d.remove_geometry(self.geometry_objects[geom_id], False)
                            self.active_geometries.remove(geom_id)
                        
                        # Add new geometry
                        self.vis3d.add_geometry(model, False)
                        self.active_geometries.add(geom_id)
                        self.geometry_objects[geom_id] = model
                        
                        # Update visualization
                        self.vis3d.update_renderer()
                except Exception as e:
                    print(f"Error creating model: {str(e)}")
            
        except Exception as e:
            print(f"Error in draw_oriented_3d_bbox: {str(e)}")

    def draw_enhanced_object_info(self, img: np.ndarray, obj: Dict, 
                                depth: float, orientation: float, 
                                x: int, y: int):
        """Draw enhanced object information including orientation"""
        # Create background for better text visibility
        orientation_deg = np.degrees(orientation) % 360
        # Shorter text format for less intrusion
        text = f"{obj['type'][0].upper()}:{depth:.1f}m"  # First letter of type + distance
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4  # Reduced font size
        thickness = 1  # Reduced thickness
        
        # Get text size for background
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw semi-transparent background with reduced size
        cv2.rectangle(img, (x-2, y-text_height-5), (x+text_width+2, y+2), 
                     (0,0,0), -1)
        
        # Draw text
        cv2.putText(img, text, (x, y), font, font_scale, obj['color'], thickness)

    def draw_scene_info(self, img: np.ndarray, detected_objects: List[Dict]):
        """Draw scene understanding information"""
        if not detected_objects:
            return
            
        # Count objects by type
        object_counts = {}
        for obj in detected_objects:
            obj_type = obj['type']
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        
        # Draw scene statistics
        y_offset = 30
        for obj_type, count in object_counts.items():
            cv2.putText(img, f"{obj_type}s: {count}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y_offset += 25

    def draw_depth_scale(self, img: np.ndarray):
        """Draw improved depth scale bar"""
        height, width = img.shape[:2]
        bar_width = 30
        bar_height = height // 2
        x = width - bar_width - 20
        y = (height - bar_height) // 2
        
        # Create gradient with improved visibility
        gradient = np.linspace(0, 255, bar_height).astype(np.uint8)
        gradient = cv2.applyColorMap(gradient[:, np.newaxis], cv2.COLORMAP_MAGMA)
        
        # Add background for better visibility
        cv2.rectangle(img, (x-35, y-10), (x+bar_width+5, y+bar_height+10), 
                     (0,0,0), -1)
        
        # Draw bar
        img[y:y+bar_height, x:x+bar_width] = gradient
        
        # Add labels with improved visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, "0m", (x-30, y+bar_height+5), 
                    font, 0.5, (255,255,255), 1)
        cv2.putText(img, "50m", (x-30, y-5), 
                    font, 0.5, (255,255,255), 1)

    def estimate_3d_dimensions(self, obj_type: str, width_2d: float, height_2d: float, depth: float) -> Tuple[float, float, float]:
        """Estimate 3D dimensions based on object type and 2D size with improved accuracy"""
        # Convert pixel sizes to meters using depth and focal length with perspective correction
        width_m = width_2d * depth / self.fx
        height_m = height_2d * depth / self.fy
        
        # Apply perspective correction factor
        perspective_factor = np.clip(1.0 - depth/100, 0.8, 1.0)  # Less aggressive scaling
        
        # Standard dimensions for common objects (in meters)
        standard_dims = {
            'car': {
                'width': 1.8,  # Average car width
                'height': 1.5, # Average car height
                'length': 4.5  # Average car length
            },
            'person': {
                'width': 0.5,  # Average person width
                'height': 1.7, # Average person height
                'length': 0.3  # Average person depth
            },
            'truck': {
                'width': 2.5,  # Average truck width
                'height': 3.0, # Average truck height
                'length': 7.0  # Average truck length
            }
        }
        
        if obj_type.lower() in standard_dims:
            std_dim = standard_dims[obj_type.lower()]
            
            # Calculate scale factor based on detected size vs standard size
            width_scale = width_m / std_dim['width']
            height_scale = height_m / std_dim['height']
            
            # Use the more reliable dimension for scaling
            scale = min(width_scale, height_scale)
            scale = np.clip(scale, 0.5, 2.0)  # Limit scaling range
            
            # Apply scale to standard dimensions
            width = std_dim['width'] * scale * perspective_factor
            height = std_dim['height'] * scale
            length = std_dim['length'] * scale * perspective_factor
            
            return (width, height, length)
        else:
            # For unknown objects, use detected dimensions with aspect ratio constraints
            aspect_ratio = width_m / height_m
            if aspect_ratio > 2.0:  # Very wide object
                length = width_m * 0.5
            elif aspect_ratio < 0.5:  # Very tall object
                length = height_m * 0.3
            else:  # Normal proportions
                length = (width_m + height_m) * 0.25
                
            return (width_m * perspective_factor, height_m, length * perspective_factor)

    def update_pose(self, R: np.ndarray = None, t: np.ndarray = None):
        """Update ego vehicle pose"""
        if R is not None and t is not None:
            try:
                # Create transformation matrix
                transform = np.eye(4)
                transform[:3, :3] = R
                
                # Scale translation for better visualization
                translation = t.ravel()
                translation_norm = np.linalg.norm(translation)
                
                if translation_norm > 0.01:  # 1cm threshold
                    transform[:3, 3] = translation
                    
                    # Update current pose
                    self.current_pose = self.current_pose @ transform
                    
                    # Add to trajectory
                    if len(self.trajectory_points) == 0 or \
                       np.linalg.norm(self.current_pose[:3, 3] - self.trajectory_points[-1]) > 0.05:
                        self.trajectory_points.append(self.current_pose[:3, 3])
                    
                    # Limit trajectory length
                    max_trajectory_points = 2000
                    if len(self.trajectory_points) > max_trajectory_points:
                        self.trajectory_points = self.trajectory_points[-max_trajectory_points:]
                
            except Exception as e:
                print(f"Error updating pose: {str(e)}")

    def estimate_depth(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate depth from a single RGB frame
        Returns: (depth_map, colored_depth)
        """
        try:
            # Store frame for later use
            self.last_frame = frame
            
            # Move frame to GPU if available
            if self.device.type == "cuda":
                frame_gpu = cv2.cuda_GpuMat()
                frame_gpu.upload(frame)
                frame_rgb = cv2.cuda.cvtColor(frame_gpu, cv2.COLOR_BGR2RGB)
                img = frame_rgb.download()
            else:
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Transform and move to GPU
            input_batch = self.transform(img).to(self.device)

            # Run inference with GPU optimization
            with torch.cuda.amp.autocast() if self.device.type == "cuda" else torch.no_grad():
                prediction = self.midas(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy()

            # Apply depth refinement
            if self.device.type == "cuda":
                depth_gpu = cv2.cuda_GpuMat()
                depth_gpu.upload(depth_map.astype(np.float32))
                depth_map = cv2.cuda.bilateralFilter(depth_gpu, 5, 75, 75).download()
            else:
                depth_map = cv2.bilateralFilter(depth_map.astype(np.float32), 5, 75, 75)

            # Scale depth map to reasonable range (in meters)
            depth_map = (depth_map / depth_map.max()) * 50  # Max depth of 50m

            # Create colored depth map
            normalized_depth = ((depth_map / 50) * 255).astype(np.uint8)
            colored_depth = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_MAGMA)

            # Store depth map for later use
            self.last_depth_map = depth_map
            
            return depth_map, colored_depth

        except Exception as e:
            print(f"Error in depth estimation: {str(e)}")
            return None, None

    def create_ground_plane_visualization(self, depth_map: np.ndarray) -> np.ndarray:
        """Create perspective ground plane visualization using depth map"""
        try:
            height, width = depth_map.shape
            ground_overlay = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Create perspective ground plane with lower horizon for more ground visibility
            horizon_y = int(height * 0.25)  # Moved horizon line up to show more ground
            
            # Create a simple dark green ground plane
            ground_mask = np.zeros((height, width, 3), dtype=np.uint8)
            ground_points = np.array([[0, height],
                                    [width, height],
                                    [width, horizon_y],
                                    [0, horizon_y]], dtype=np.int32)
            cv2.fillPoly(ground_mask, [ground_points], (0, 40, 0))  # Dark green ground
            
            # Add the ground mask to the overlay
            ground_overlay = cv2.addWeighted(ground_overlay, 0.7, ground_mask, 0.3, 0)
            
            return ground_overlay
            
        except Exception as e:
            print(f"Error creating ground plane: {str(e)}")
            return None

    def estimate_pose_essential(self, pts1: np.ndarray, pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate relative pose using essential matrix decomposition
        Returns: R, t, mask of inliers
        """
        # Normalize points using camera intrinsics
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.camera_matrix, self.dist_coeffs)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.camera_matrix, self.dist_coeffs)

        # Estimate essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            pts1_norm, pts2_norm,
            focal=1.0, pp=(0., 0.),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.001
        )

        if E is None:
            return np.eye(3), np.zeros((3, 1)), None

        # Recover pose from essential matrix
        _, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm, mask=mask)
        return R, t, mask

    def triangulate_points_robust(self, pts1: np.ndarray, pts2: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points with robust outlier rejection
        """
        # Create projection matrices
        P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = np.hstack((R, t))

        # Normalize points
        pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2), self.camera_matrix, self.dist_coeffs)
        pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2), self.camera_matrix, self.dist_coeffs)

        # Triangulate
        points4D = cv2.triangulatePoints(P1, P2, pts1_norm.reshape(2, -1), pts2_norm.reshape(2, -1))
        points3D = (points4D[:3] / points4D[3]).T

        # Filter points based on reprojection error and depth
        valid_points = []
        for i, pt3D in enumerate(points3D):
            # Check depth
            if pt3D[2] <= 0:
                continue

            # Project point
            pt1_proj = self.camera_matrix @ (P1 @ np.append(pt3D, 1))
            pt2_proj = self.camera_matrix @ (P2 @ np.append(pt3D, 1))
            pt1_proj = pt1_proj[:2] / pt1_proj[2]
            pt2_proj = pt2_proj[:2] / pt2_proj[2]

            # Calculate reprojection error
            error1 = np.linalg.norm(pt1_proj - pts1[i])
            error2 = np.linalg.norm(pt2_proj - pts2[i])

            if error1 < 5.0 and error2 < 5.0:  # 5 pixel threshold
                valid_points.append(pt3D)

        return np.array(valid_points)

    def bundle_adjustment(self, points3D: np.ndarray, points2D: List[np.ndarray], 
                        camera_poses: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Optimize 3D points and camera poses using bundle adjustment
        """
        def objective(params):
            n_points = len(points3D)
            n_poses = len(camera_poses)
            
            # Extract parameters
            points_new = params[:n_points*3].reshape((n_points, 3))
            poses_new = params[n_points*3:].reshape((n_poses, 6))
            
            # Compute reprojection error
            total_error = []
            
            for i, pose in enumerate(poses_new):
                R = cv2.Rodrigues(pose[:3])[0]
                t = pose[3:].reshape(3, 1)
                P = np.hstack((R, t))
                
                # Project points
                projected = (self.camera_matrix @ P @ np.hstack((points_new, np.ones((n_points, 1))).T).T)
                projected = projected[:, :2] / projected[:, 2:]
                
                # Compute error
                error = (projected - points2D[i]).ravel()
                total_error.extend(error)
            
            return np.array(total_error)

        # Initialize parameters
        n_points = len(points3D)
        n_poses = len(camera_poses)
        params = np.zeros(n_points*3 + n_poses*6)
        
        # Fill initial parameters
        params[:n_points*3] = points3D.ravel()
        for i, pose in enumerate(camera_poses):
            R = pose[:3, :3]
            t = pose[:3, 3]
            rvec = cv2.Rodrigues(R)[0].ravel()
            params[n_points*3 + i*6:n_points*3 + (i+1)*6] = np.concatenate([rvec, t])

        # Optimize
        result = least_squares(objective, params, method='lm', max_nfev=100)
        
        # Extract optimized parameters
        points_optimized = result.x[:n_points*3].reshape((n_points, 3))
        poses_optimized = []
        for i in range(n_poses):
            pose_params = result.x[n_points*3 + i*6:n_points*3 + (i+1)*6]
            R = cv2.Rodrigues(pose_params[:3])[0]
            t = pose_params[3:]
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            poses_optimized.append(pose)

        return points_optimized, poses_optimized

    def create_dense_reconstruction(self, depth_map: np.ndarray, color_image: np.ndarray, 
                                  pose: np.ndarray) -> o3d.geometry.PointCloud:
        """
        Create dense point cloud from depth map and color image
        """
        height, width = depth_map.shape
        points = []
        colors = []

        for v in range(height):
            for u in range(width):
                depth = depth_map[v, u]
                if depth > 0:
                    # Back-project to 3D
                    x = (u - self.cx) * depth / self.fx
                    y = (v - self.cy) * depth / self.fy
                    z = depth

                    # Transform to world coordinates
                    point = pose[:3, :3] @ np.array([x, y, z]) + pose[:3, 3]
                    points.append(point)
                    colors.append(color_image[v, u] / 255.0)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd, _ = pcd.remove_radius_outlier(nb_points=16, radius=0.05)

        return pcd

    def update_3d_reconstruction(self, frame: np.ndarray, depth_map: np.ndarray):
        """Update 3D reconstruction with new frame"""
        try:
            # Detect features
            keypoints, descriptors = self.feature_detector.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None)
            
            if len(self.keyframes) > 0:
                # Match with last keyframe
                matches = self.feature_matcher.match(self.keyframe_descriptors[-1], descriptors)
                matches = sorted(matches, key=lambda x: x.distance)[:100]

                if len(matches) >= 8:
                    # Extract matched points
                    pts1 = np.float32([self.keyframe_keypoints[-1][m.queryIdx].pt for m in matches])
                    pts2 = np.float32([keypoints[m.trainIdx].pt for m in matches])

                    # Estimate relative pose
                    R, t, mask = self.estimate_pose_essential(pts1, pts2)
                    
                    if mask is not None:
                        # Update global pose
                        current_pose = np.eye(4)
                        current_pose[:3, :3] = R
                        current_pose[:3, 3] = t.ravel()
                        self.current_pose = self.current_pose @ current_pose

                        # Triangulate points
                        points3D = self.triangulate_points_robust(pts1[mask.ravel() == 1], 
                                                                pts2[mask.ravel() == 1], R, t)

                        # Create dense reconstruction
                        dense_pcd = self.create_dense_reconstruction(depth_map, frame, self.current_pose)

                        # Update visualization
                        if len(points3D) > 0:
                            # Add new points to global point cloud
                            self.point_cloud.extend(points3D.tolist())
                            colors = []
                            for pt in pts2[mask.ravel() == 1]:
                                y, x = int(pt[1]), int(pt[0])
                                if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                                    colors.append(frame[y, x][::-1] / 255.0)  # BGR to RGB
                                else:
                                    colors.append(np.array([1.0, 0.0, 0.0]))  # Red for invalid points
                            self.point_colors.extend(colors)

                            # Create and update sparse point cloud
                            sparse_pcd = o3d.geometry.PointCloud()
                            sparse_pcd.points = o3d.utility.Vector3dVector(np.array(self.point_cloud))
                            sparse_pcd.colors = o3d.utility.Vector3dVector(np.array(self.point_colors))
                            
                            # Remove outliers from sparse reconstruction
                            sparse_pcd, _ = sparse_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                            
                            # Merge sparse and dense reconstructions
                            if dense_pcd is not None:
                                combined_pcd = sparse_pcd + dense_pcd
                                
                                # Voxel downsampling to reduce point density
                                combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.05)
                                
                                # Update visualizer
                                self.vis3d.clear_geometries()
                                
                                # Add coordinate frame
                                coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
                                self.vis3d.add_geometry(coord_frame)
                                
                                # Add point cloud
                                self.vis3d.add_geometry(combined_pcd)
                                
                                # Add camera trajectory
                                if len(self.keyframe_poses) > 1:
                                    points = []
                                    lines = []
                                    colors = []
                                    for i in range(len(self.keyframe_poses)-1):
                                        points.append(self.keyframe_poses[i][:3, 3])
                                        points.append(self.keyframe_poses[i+1][:3, 3])
                                        lines.append([2*i, 2*i+1])
                                        colors.append([1, 0, 0])  # Red color for trajectory
                                    
                                    line_set = o3d.geometry.LineSet()
                                    line_set.points = o3d.utility.Vector3dVector(np.array(points))
                                    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
                                    line_set.colors = o3d.utility.Vector3dVector(np.array(colors))
                                    self.vis3d.add_geometry(line_set)
                                
                                # Update camera viewpoint
                                ctr = self.vis3d.get_view_control()
                                ctr.set_zoom(0.3)
                                ctr.set_front([0, 0, -1])
                                ctr.set_up([0, -1, 0])
                                
                                # Update visualization
                                self.vis3d.poll_events()
                                self.vis3d.update_renderer()

            # Store keyframe
            if len(self.keyframes) == 0 or len(matches) < 50:
                self.keyframes.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
                self.keyframe_keypoints.append(keypoints)
                self.keyframe_descriptors.append(descriptors)
                self.keyframe_poses.append(self.current_pose.copy())

                # Limit number of keyframes
                max_keyframes = 20
                if len(self.keyframes) > max_keyframes:
                    self.keyframes = self.keyframes[-max_keyframes:]
                    self.keyframe_keypoints = self.keyframe_keypoints[-max_keyframes:]
                    self.keyframe_descriptors = self.keyframe_descriptors[-max_keyframes:]
                    self.keyframe_poses = self.keyframe_poses[-max_keyframes:]
                    
        except Exception as e:
            print(f"Error in 3D reconstruction: {str(e)}")

    def load_3d_models(self):
        """Load and cache 3D models for detected objects"""
        try:
            # Define model paths
            model_paths = {
                'person': 'models/pedestrians/person.obj',
                'car': 'models/vehicles/car.obj',
                'truck': 'models/vehicles/truck.obj',
                'bus': 'models/vehicles/bus.obj',
                'stop_sign': 'models/signs/stop_sign.obj',
                'traffic_light': 'models/signs/traffic_light.obj'
            }
            
            # Load each model
            for obj_type, path in model_paths.items():
                if os.path.exists(path):
                    try:
                        mesh = o3d.io.read_triangle_mesh(path)
                        # Center the model and normalize scale
                        mesh.compute_vertex_normals()
                        mesh.compute_triangle_normals()
                        center = mesh.get_center()
                        mesh.translate(-center)
                        
                        # Scale to unit size (will be scaled according to detection later)
                        vertices = np.asarray(mesh.vertices)
                        max_dim = np.max(vertices.max(axis=0) - vertices.min(axis=0))
                        scale = 1.0 / max_dim
                        mesh.scale(scale, center=[0, 0, 0])
                        
                        self.model_cache[obj_type] = mesh
                        print(f"Loaded 3D model for {obj_type}")
                    except Exception as e:
                        print(f"Error loading model {obj_type}: {str(e)}")
            
        except Exception as e:
            print(f"Error in load_3d_models: {str(e)}")

    def place_3d_model(self, obj_type: str, bbox: List[int], depth: float, orientation: float) -> Optional[o3d.geometry.TriangleMesh]:
        """Place and orient 3D model based on detection"""
        try:
            if obj_type not in self.model_cache:
                return None
                
            # Create a deep copy of the model by copying vertices and triangles
            original_mesh = self.model_cache[obj_type]
            model = o3d.geometry.TriangleMesh()
            model.vertices = o3d.utility.Vector3dVector(np.asarray(original_mesh.vertices))
            model.triangles = o3d.utility.Vector3iVector(np.asarray(original_mesh.triangles))
            if original_mesh.has_vertex_normals():
                model.vertex_normals = o3d.utility.Vector3dVector(
                    np.asarray(original_mesh.vertex_normals))
            if original_mesh.has_vertex_colors():
                model.vertex_colors = o3d.utility.Vector3dVector(
                    np.asarray(original_mesh.vertex_colors))
            if original_mesh.has_triangle_normals():
                model.triangle_normals = o3d.utility.Vector3dVector(
                    np.asarray(original_mesh.triangle_normals))
            
            # Calculate 3D position
            x1, y1, x2, y2 = bbox
            center_x = ((x1 + x2) / 2 - self.cx) * depth / self.fx
            center_y = ((y1 + y2) / 2 - self.cy) * depth / self.fy
            center_z = depth
            
            # Calculate scale based on bounding box and depth
            target_scale = self.model_scales.get(obj_type, 1.0)
            bbox_height = y2 - y1
            pixel_to_meter = (bbox_height * depth / self.fy) / target_scale
            scale_factor = target_scale * pixel_to_meter
            
            # Apply transformations
            model.scale(scale_factor, center=[0, 0, 0])
            
            # Create rotation matrix for orientation
            R = np.array([
                [np.cos(orientation), 0, np.sin(orientation)],
                [0, 1, 0],
                [-np.sin(orientation), 0, np.cos(orientation)]
            ])
            
            # Apply rotation and translation
            model.rotate(R, center=[0, 0, 0])
            model.translate([center_x, center_y, center_z])
            
            return model
            
        except Exception as e:
            print(f"Error placing 3D model: {str(e)}")
            return None