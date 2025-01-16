import numpy as np
import cv2
import open3d as o3d
from typing import Tuple, List, Dict
import os
from ultralytics import YOLO
import torch
import time


class AutonomousPerception:
    def __init__(self, kitti_data_path: str = None):
        """Initialize autonomous perception system"""
        # Initialize YOLO model with fastest version
        try:
            # Use YOLOv8n.pt (nano) - fastest version
            self.detector = YOLO('yolov8n.pt')

            # Enable GPU if available
            if torch.cuda.is_available():
                print("Using GPU for detection")
                self.device = 'cuda'
            else:
                print("Using CPU for detection")
                self.device = 'cpu'

            # Optimize parameters for speed
            self.detector.conf = 0.35  # Higher confidence threshold to reduce false positives
            self.detector.iou = 0.35   # Lower IoU for faster NMS
            self.detector.agnostic = True  # Class-agnostic NMS for speed
            self.detector.max_det = 20  # Limit detections for speed
            self.detector.verbose = False  # Disable verbose output

        except Exception as e:
            print(f"Error loading YOLO model: {str(e)}")
            print("Please install ultralytics: pip install ultralytics")
            raise

        # Camera parameters
        self.camera_intrinsics = np.array([
            [718.856, 0, 607.1928],
            [0, 718.856, 185.2157],
            [0, 0, 1]
        ], dtype=np.float64)

        if kitti_data_path:
            self.load_kitti_calibration(kitti_data_path)

        # Initialize visualization
        self.use_3d_vis = False
        self.vis = None
        self.init_3d_visualization()

        # Reduce feature detection for speed
        self.feature_detector = cv2.SIFT_create(
            nfeatures=1000)  # Reduced from 3000
        self.feature_matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None
        self.prev_points_3d = []

        # Simplified class mapping for speed (only essential classes)
        self.yolo_classes = {
            2: {'name': 'car', 'color': (0, 140, 255)},      # Orange for cars
            # Yellow for people
            0: {'name': 'person', 'color': (0, 255, 255)},
            # Orange-red for buses
            5: {'name': 'bus', 'color': (0, 165, 255)},
            # Dark orange for trucks
            7: {'name': 'truck', 'color': (0, 100, 255)},
            # Green for traffic lights
            9: {'name': 'traffic_light', 'color': (0, 255, 0)},
            # Red for stop signs
            11: {'name': 'stop_sign', 'color': (0, 0, 255)},
        }

    def load_kitti_calibration(self, kitti_path: str):
        """Load KITTI camera calibration"""
        calib_file = os.path.join(kitti_path, 'calib_cam_to_cam.txt')
        if not os.path.exists(calib_file):
            print(f"Warning: Calibration file not found at {calib_file}")
            print("Using default KITTI calibration values...")
            return

        try:
            with open(calib_file, 'r') as f:
                for line in f.readlines():
                    if line.startswith('P0:'):
                        P = np.array([float(x)
                                     for x in line.split()[1:]]).reshape(3, 4)
                        self.camera_intrinsics = P[:3, :3].astype(np.float64)
                        print("Successfully loaded camera calibration:")
                        print(self.camera_intrinsics)
                        return

            print("Warning: Could not find P0 matrix in calibration file")
            print("Using default KITTI calibration values...")
        except Exception as e:
            print(f"Warning: Error reading calibration file: {str(e)}")
            print("Using default KITTI calibration values...")

    def detect_lanes(self, frame: np.ndarray) -> np.ndarray:
        """Detect lanes using improved computer vision techniques"""
        # Convert to grayscale and enhance contrast
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply blur and edge detection
        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Create ROI mask (trapezoid shape for better lane focus)
        height, width = edges.shape
        roi_vertices = np.array([
            [(0, height),
             (width * 0.45, height * 0.6),  # Left vanishing point
             (width * 0.55, height * 0.6),  # Right vanishing point
             (width, height)]
        ], dtype=np.int32)

        # Apply ROI mask
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Detect lines using probabilistic Hough transform with dynamic parameters
        min_line_length = height * 0.3  # Minimum 30% of image height
        max_line_gap = height * 0.05    # Maximum 5% gap
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=50,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap
        )

        # Create lane mask
        lane_mask = np.zeros_like(frame)

        if lines is not None:
            # Separate left and right lanes based on slope
            left_lines = []
            right_lines = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 == 0:  # Avoid division by zero
                    continue

                slope = (y2 - y1) / (x2 - x1)

                # Filter by slope
                if 0.3 < abs(slope) < 2.0:  # Reasonable lane slope range
                    if slope < 0:  # Left lane
                        left_lines.append(line[0])
                    else:  # Right lane
                        right_lines.append(line[0])

            # Average and extrapolate lanes
            def average_lane(lines):
                if not lines:
                    return None

                avg_line = np.mean(lines, axis=0, dtype=np.int32)
                x1, y1, x2, y2 = avg_line

                # Extrapolate to bottom of image
                if x2 - x1 != 0:
                    slope = (y2 - y1) / (x2 - x1)
                    b = y1 - slope * x1

                    # Bottom point
                    bottom_x = int((height - b) / slope)
                    # Top point (60% of image height)
                    top_x = int((height * 0.6 - b) / slope)

                    return np.array([
                        [bottom_x, height],
                        [top_x, int(height * 0.6)]
                    ])
                return None

            # Draw averaged lanes
            left_avg = average_lane(left_lines)
            right_avg = average_lane(right_lines)

            if left_avg is not None:
                cv2.line(lane_mask, tuple(left_avg[0]), tuple(
                    left_avg[1]), (0, 255, 0), 5)
            if right_avg is not None:
                cv2.line(lane_mask, tuple(right_avg[0]), tuple(
                    right_avg[1]), (0, 255, 0), 5)

            # Update driving corridor based on detected lanes
            if left_avg is not None and right_avg is not None:
                self.update_driving_corridor(
                    left_avg, right_avg, height, width)

        return lane_mask

    def update_driving_corridor(self, left_lane, right_lane, height, width):
        """Update driving corridor based on detected lanes"""
        # Calculate corridor points based on lane detections
        left_bottom, left_top = left_lane
        right_bottom, right_top = right_lane

        # Calculate middle points for smoother corridor
        mid_bottom_x = (left_bottom[0] + right_bottom[0]) // 2
        mid_top_x = (left_top[0] + right_top[0]) // 2

        # Update corridor points with some margin
        margin = width * 0.05  # 5% margin
        self.corridor_points = np.array([
            [(left_bottom[0] + margin, height),
             (left_top[0] + margin, int(height * 0.6)),
             (right_top[0] - margin, int(height * 0.6)),
             (right_bottom[0] - margin, height)]], dtype=np.int32)

    def detect_features(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """Detect features in frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.feature_detector.detectAndCompute(
            gray, None)
        return keypoints, descriptors

    def estimate_motion(self, keypoints1, keypoints2, matches) -> Tuple[np.ndarray, np.ndarray]:
        """Estimate camera motion between frames with improved robustness"""
        if len(matches) < 8:
            return np.eye(3), np.zeros((3, 1))

        # Extract matched points
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        # Normalize points using camera intrinsics
        pts1_norm = cv2.undistortPoints(
            pts1.reshape(-1, 1, 2), self.camera_intrinsics, None)
        pts2_norm = cv2.undistortPoints(
            pts2.reshape(-1, 1, 2), self.camera_intrinsics, None)

        # Essential matrix estimation with stricter RANSAC
        E, mask = cv2.findEssentialMat(
            pts1_norm, pts2_norm,
            focal=1.0, pp=(0., 0.),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=0.001
        )

        if E is None:
            return np.eye(3), np.zeros((3, 1))

        # Recover pose with more points
        _, R, t, mask = cv2.recoverPose(E, pts1_norm, pts2_norm, mask=mask)

        return R, t

    def triangulate_points(self, keypoints1, keypoints2, matches, R, t) -> np.ndarray:
        """Triangulate 3D points with improved accuracy"""
        # Extract matched points
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

        # Normalize points
        pts1_norm = cv2.undistortPoints(
            pts1.reshape(-1, 1, 2), self.camera_intrinsics, None)
        pts2_norm = cv2.undistortPoints(
            pts2.reshape(-1, 1, 2), self.camera_intrinsics, None)

        # Projection matrices
        P1 = np.eye(3, 4)  # First camera matrix
        P2 = np.hstack((R, t))  # Second camera matrix

        # Triangulate points
        points4D = cv2.triangulatePoints(
            P1, P2, pts1_norm.reshape(2, -1), pts2_norm.reshape(2, -1))
        points3D = (points4D[:3] / points4D[3]).T

        return points3D

    def create_3d_bounding_box(self, bbox: List[int], obj_type: str, color: tuple) -> o3d.geometry.LineSet:
        """Create a 3D bounding box using Open3D"""
        x1, y1, x2, y2 = bbox

        # Get object dimensions based on type (in meters)
        if obj_type in ['car', 'bus', 'truck']:
            width = 2.0
            height = 1.6
            length = 4.0
        elif obj_type == 'person':
            width = 0.6
            height = 1.8
            length = 0.6
        else:  # signs and other objects
            width = 0.8
            height = 0.8
            length = 0.3

        # Calculate object position in 3D space
        # Convert image coordinates to 3D world coordinates
        # Approximate image height
        image_height = self.camera_intrinsics[1, 2] * 2
        depth = 30 * (1 - y2 / image_height)  # Depth estimation

        # Use camera intrinsics to get 3D position
        center_x = (
            (x1 + x2) / 2 - self.camera_intrinsics[0, 2]) * depth / self.camera_intrinsics[0, 0]
        center_y = (
            (y1 + y2) / 2 - self.camera_intrinsics[1, 2]) * depth / self.camera_intrinsics[1, 1]
        center_z = depth

        # Create box vertices
        vertices = [
            [center_x - width/2, center_y - height/2,
                center_z - length/2],  # Front bottom left
            [center_x + width/2, center_y - height/2,
                center_z - length/2],  # Front bottom right
            [center_x + width/2, center_y + height/2,
                center_z - length/2],  # Front top right
            [center_x - width/2, center_y + height/2,
                center_z - length/2],  # Front top left
            [center_x - width/2, center_y - height/2,
                center_z + length/2],  # Back bottom left
            [center_x + width/2, center_y - height/2,
                center_z + length/2],  # Back bottom right
            [center_x + width/2, center_y + height/2,
                center_z + length/2],  # Back top right
            [center_x - width/2, center_y + height/2,
                center_z + length/2]   # Back top left
        ]

        # Define edges connecting vertices
        lines = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # Front face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Back face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting lines
        ]

        # Create Open3D LineSet
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(vertices),
            lines=o3d.utility.Vector2iVector(lines)
        )

        # Set color (convert from BGR to RGB)
        rgb_color = (color[2]/255, color[1]/255, color[0]/255)
        line_set.paint_uniform_color(rgb_color)

        return line_set

    def init_3d_visualization(self):
        """Initialize visualization windows and Open3D visualizer"""
        try:
            # Create OpenCV windows
            screen_width = 1920  # Default resolution
            screen_height = 1080
            window_width = screen_width // 2
            window_height = screen_height // 2

            cv2.namedWindow("Autonomous Vehicle View", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Feature Matching", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Info View", cv2.WINDOW_NORMAL)

            # Position OpenCV windows
            cv2.resizeWindow("Autonomous Vehicle View",
                             window_width, window_height)
            cv2.resizeWindow("Feature Matching", window_width, window_height)
            cv2.resizeWindow("Info View", window_width, window_height)

            cv2.moveWindow("Autonomous Vehicle View", 0, 0)
            cv2.moveWindow("Feature Matching", window_width, 0)
            cv2.moveWindow("Info View", 0, window_height)

            # Initialize Open3D visualizer
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window(
                "3D Scene", width=window_width, height=window_height)

            # Set up initial view
            view_control = self.vis.get_view_control()
            view_control.set_zoom(0.3)
            view_control.set_front([0, 0, -1])
            view_control.set_up([0, -1, 0])

            # Add coordinate frame
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=2.0)
            self.vis.add_geometry(coordinate_frame)

            # Create ground plane
            self.create_ground_plane()

            self.use_3d_vis = True
            print("3D visualization initialized successfully")

            # Initialize tracking variables
            self.current_geometries = {}  # Track current objects in scene
            self.last_update_time = time.time()

        except Exception as e:
            print(f"\nError initializing visualization: {str(e)}")
            self.use_3d_vis = False

    def create_ground_plane(self):
        """Create and add ground plane to visualizer"""
        # Create ground plane mesh
        plane_mesh = o3d.geometry.TriangleMesh.create_box(40, 0.1, 40)
        plane_mesh.translate([-20, -0.05, -20])  # Center the plane
        plane_mesh.paint_uniform_color([0.2, 0.2, 0.2])  # Dark gray

        # Add to visualizer
        self.vis.add_geometry(plane_mesh)
        self.ground_plane = plane_mesh

    def detect_objects(self, frame: np.ndarray) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Detect objects using YOLOv8n with speed optimizations"""
        vehicles = []
        pedestrians = []
        signs = []

        try:
            # Resize frame for faster processing (optional, uncomment if needed)
            # frame = cv2.resize(frame, (640, 480))

            # Run YOLOv8 detection without tracking for speed
            results = self.detector(frame, verbose=False)

            if results and len(results) > 0:
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())

                        # Skip if class not in our mapping
                        if class_id not in self.yolo_classes:
                            continue

                        # Get object info
                        obj_info = self.yolo_classes[class_id]
                        obj_type = obj_info['name']
                        color = obj_info['color']

                        # Create detection object (simplified for speed)
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'type': obj_type,
                            'color': color
                        }

                        # Simplified categorization with adjusted thresholds
                        if obj_type in ['car', 'bus', 'truck']:
                            if confidence > 0.4:  # Reduced threshold
                                vehicles.append(detection)
                        elif obj_type == 'person':
                            if confidence > 0.35:  # Reduced threshold
                                pedestrians.append(detection)
                        elif obj_type in ['stop_sign', 'traffic_light']:
                            if confidence > 0.3:  # Reduced threshold
                                signs.append(detection)

        except Exception as e:
            print(f"Warning: YOLO detection failed: {str(e)}")

        return vehicles, pedestrians, signs

    def _non_max_suppression(self, boxes, weights, overlapThresh):
        """Apply non-maximum suppression to avoid duplicate detections"""
        if len(boxes) == 0:
            return []

        # Convert boxes to format [x1, y1, x2, y2]
        boxes_array = np.array([[x, y, x + w, y + h]
                               for (x, y, w, h) in boxes])

        # Compute areas
        x1 = boxes_array[:, 0]
        y1 = boxes_array[:, 1]
        x2 = boxes_array[:, 2]
        y2 = boxes_array[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        # Sort by confidence
        idxs = np.argsort(weights)[::-1]
        pick = []

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # Find overlapping boxes
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # Compute overlap
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]

            # Delete overlapping boxes
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        return pick

    def update_visualization(self, frame: np.ndarray, points_3d: np.ndarray,
                             lane_mask: np.ndarray, vehicles: List[Dict],
                             pedestrians: List[Dict], signs: List[Dict]):
        """Update visualization with dynamic 3D bounding boxes"""
        if not self.use_3d_vis:
            return

        try:
            # Update 2D visualization
            vis_frame = frame.copy()
            height, width = frame.shape[:2]

            # Draw detected lanes and corridor
            if lane_mask is not None:
                cv2.addWeighted(lane_mask, 0.4, vis_frame, 1, 0, vis_frame)
            if hasattr(self, 'corridor_points'):
                overlay = vis_frame.copy()
                cv2.fillPoly(overlay, self.corridor_points, (0, 255, 0))
                cv2.addWeighted(overlay, 0.2, vis_frame, 0.8, 0, vis_frame)

            # Update 3D visualization
            # Clear previous objects
            for obj_id in list(self.current_geometries.keys()):
                self.vis.remove_geometry(
                    self.current_geometries[obj_id], False)
            self.current_geometries.clear()

            # Sort objects by distance
            all_objects = vehicles + pedestrians + signs
            all_objects.sort(key=lambda x: x['bbox'][3], reverse=True)

            # Process each object
            for i, obj in enumerate(all_objects):
                bbox = obj['bbox']
                color = obj['color']
                obj_type = obj['type']

                # Draw 2D bounding box and label
                x1, y1, x2, y2 = [int(x) for x in bbox]
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

                distance = 30 * (1 - y2 / height)
                label = f"{obj_type}: {distance:.1f}m"
                cv2.putText(vis_frame, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Create and add 3D bounding box
                box_3d = self.create_3d_bounding_box(bbox, obj_type, color)
                self.vis.add_geometry(box_3d, False)
                self.current_geometries[f"obj_{i}"] = box_3d

            # Update point cloud if available
            if points_3d is not None and len(points_3d) > 0:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_3d)
                pcd.paint_uniform_color([0.7, 0.7, 0.7])  # Light gray
                self.vis.add_geometry(pcd, False)
                self.current_geometries["point_cloud"] = pcd

            # Update Open3D visualization
            self.vis.poll_events()
            self.vis.update_renderer()

            # Show 2D visualizations
            cv2.imshow("Autonomous Vehicle View", vis_frame)

            # Add stats overlay
            cv2.putText(vis_frame, f"Vehicles: {len(vehicles)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
            cv2.putText(vis_frame, f"People: {len(pedestrians)}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(vis_frame, f"Signs: {len(signs)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        except Exception as e:
            print(f"\nWarning: Visualization update failed: {str(e)}")
            import traceback
            traceback.print_exc()
            self.use_3d_vis = False

    def process_frame(self, frame: np.ndarray):
        """Process a single frame with enhanced detection and visualization"""
        # Detect lanes
        lane_mask = self.detect_lanes(frame)

        # Detect objects
        vehicles, pedestrians, signs = self.detect_objects(frame)

        # Detect features and estimate motion
        keypoints, descriptors = self.detect_features(frame)
        points_3d = None
        R = None
        t = None

        if self.prev_frame is not None and self.prev_keypoints is not None:
            # Match features
            matches = self.feature_matcher.match(
                self.prev_descriptors, descriptors)

            # Draw feature matches
            matches_img = cv2.drawMatches(self.prev_frame, self.prev_keypoints,
                                          frame, keypoints, matches[:100], None,
                                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            cv2.imshow("Feature Matching", matches_img)

            # Filter good matches
            good_matches = sorted(matches, key=lambda x: x.distance)[:100]

            if len(good_matches) >= 8:
                # Estimate motion
                R, t = self.estimate_motion(
                    self.prev_keypoints, keypoints, good_matches)

                # Triangulate points
                points_3d = self.triangulate_points(
                    self.prev_keypoints, keypoints, good_matches, R, t)

        # Update previous frame data
        self.prev_frame = frame.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        # Update visualization with all detections and SLAM data
        self.update_visualization(
            frame, points_3d, lane_mask, vehicles, pedestrians, signs)

        # Create info view
        info_frame = frame.copy()

        # Draw all detections in info view
        for obj in vehicles + pedestrians + signs:
            bbox = obj['bbox']
            cv2.rectangle(info_frame,
                          (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])),
                          obj['color'], 2)

        # Add text overlays
        cv2.putText(info_frame, f"Vehicles: {len(vehicles)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(info_frame, f"Pedestrians: {len(pedestrians)}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(info_frame, f"Signs: {len(signs)}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Show info view
        cv2.imshow("Info View", info_frame)

        return info_frame
