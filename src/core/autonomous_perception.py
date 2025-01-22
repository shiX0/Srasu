import numpy as np
import cv2
import open3d as o3d
from typing import Tuple, List, Dict
import os
from ultralytics import YOLO
import torch


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

    def init_3d_visualization(self):
        """Initialize visualization windows and position them"""
        try:
            # Get screen resolution
            screen_width = 1920  # Default resolution, adjust if needed
            screen_height = 1080

            # Create windows
            cv2.namedWindow("Autonomous Vehicle View", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Feature Matching", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Info View", cv2.WINDOW_NORMAL)
            cv2.namedWindow("SLAM Visualization", cv2.WINDOW_NORMAL)

            # Set window sizes (each taking a quarter of the screen)
            window_width = screen_width // 2
            window_height = screen_height // 2

            # Position windows in corners
            cv2.resizeWindow("Autonomous Vehicle View",
                             window_width, window_height)
            cv2.resizeWindow("Feature Matching", window_width, window_height)
            cv2.resizeWindow("Info View", window_width, window_height)
            cv2.resizeWindow("SLAM Visualization", window_width, window_height)

            # Top-left corner
            cv2.moveWindow("Autonomous Vehicle View", 0, 0)
            # Top-right corner
            cv2.moveWindow("Feature Matching", window_width, 0)
            # Bottom-left corner
            cv2.moveWindow("Info View", 0, window_height)
            # Bottom-right corner
            cv2.moveWindow("SLAM Visualization", window_width, window_height)

            self.use_3d_vis = True
            print("Visualization initialized successfully")

            # Initialize SLAM visualization elements
            self.slam_points = []
            self.camera_trajectory = []
            self.current_pose = np.eye(4)
            self.slam_colors = []

        except Exception as e:
            print(f"\nError initializing visualization: {str(e)}")
            self.use_3d_vis = False

    def update_slam_visualization(self, points_3d: np.ndarray, R: np.ndarray, t: np.ndarray):
        """Update SLAM visualization to match reference style"""
        try:
            # Create black background
            vis_img = np.zeros((800, 800, 3), dtype=np.uint8)

            # Default scale and center values
            scale = 50.0
            x_center = 0
            z_center = 0

            # Update camera trajectory
            if R is not None and t is not None:
                # Update current pose with scale factor for better visualization
                transform = np.eye(4)
                transform[:3, :3] = R
                # Scale up translation for better visibility
                transform[:3, 3] = t.ravel() * 5
                self.current_pose = self.current_pose @ transform

                # Add camera position if movement is significant
                if np.linalg.norm(t) > 0.01:  # Minimum movement threshold
                    self.camera_trajectory.append(self.current_pose[:3, 3])

            # Process new 3D points
            if points_3d is not None and len(points_3d) > 0:
                valid_points = []
                for point in points_3d:
                    # Stricter filtering for better point cloud quality
                    if (np.all(np.isfinite(point)) and
                        np.linalg.norm(point) < 100 and  # Increased range
                            abs(point[1]) < 20):  # Increased height range
                        # Transform point to world coordinates
                        world_point = self.current_pose[:3,
                                                        :3] @ point + self.current_pose[:3, 3]
                        valid_points.append(world_point)
                        # Color based on height (green gradient)
                        height_norm = np.clip((world_point[1] + 20) / 40, 0, 1)
                        self.slam_colors.append([0, int(255 * height_norm), 0])

                self.slam_points.extend(valid_points)

            # Keep only recent points for better visualization
            max_points = 50000  # Increased point limit
            if len(self.slam_points) > max_points:
                self.slam_points = self.slam_points[-max_points:]
                self.slam_colors = self.slam_colors[-max_points:]

            if len(self.slam_points) > 0:
                points = np.array(self.slam_points)

                # Dynamic scale calculation
                x_spread = np.ptp(points[:, 0])
                z_spread = np.ptp(points[:, 2])
                if x_spread > 0 and z_spread > 0:
                    scale = min(700 / max(x_spread, z_spread, 1),
                                200)  # Increased max scale

                # Center calculation using percentile for robustness
                x_center = np.percentile(points[:, 0], 50)
                z_center = np.percentile(points[:, 2], 50)

                # Project points to 2D
                points_2d = points[:, [0, 2]]
                points_2d[:, 0] = (points_2d[:, 0] - x_center) * scale + 400
                points_2d[:, 1] = (points_2d[:, 1] - z_center) * scale + 400
                points_2d = np.clip(points_2d, 0, 799).astype(np.int32)

                # Draw points with increased density
                for pt, color in zip(points_2d, self.slam_colors):
                    cv2.circle(vis_img, tuple(pt), 1, color, -1)

                # Draw trajectory
                if len(self.camera_trajectory) > 1:
                    trajectory = np.array(self.camera_trajectory)
                    traj_2d = trajectory[:, [0, 2]]
                    traj_2d[:, 0] = (traj_2d[:, 0] - x_center) * scale + 400
                    traj_2d[:, 1] = (traj_2d[:, 1] - z_center) * scale + 400
                    traj_2d = np.clip(traj_2d, 0, 799).astype(np.int32)

                    # Draw trajectory line in red
                    for i in range(len(traj_2d) - 1):
                        cv2.line(vis_img, tuple(traj_2d[i]), tuple(traj_2d[i + 1]),
                                 (0, 0, 255), 2)

            # Add grid (fainter than in original)
            grid_spacing = 50
            grid_color = (20, 20, 20)  # Darker grid
            for i in range(0, 800, grid_spacing):
                cv2.line(vis_img, (i, 0), (i, 800), grid_color, 1)
                cv2.line(vis_img, (0, i), (800, i), grid_color, 1)

            # Add information overlay (matching reference style)
            cv2.putText(vis_img, f"Points: {len(self.slam_points)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(vis_img, f"Scale: {10/scale:.1f}m/pixel", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add orientation compass (smaller and more subtle)
            compass_center = (750, 50)
            compass_size = 20
            cv2.circle(vis_img, compass_center, compass_size, (50, 50, 50), 1)
            cv2.line(vis_img, compass_center,
                     (compass_center[0], compass_center[1] - compass_size), (0, 0, 255), 1)
            cv2.putText(vis_img, "N", (compass_center[0] - 5, compass_center[1] - compass_size - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Show visualization
            cv2.imshow("SLAM Visualization", vis_img)

        except Exception as e:
            print(f"Warning: SLAM visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()

    def create_ground_plane(self):
        """Create road surface with lane markings"""
        # Create road surface (darker for better contrast)
        road_width = 20
        road_length = 40
        plane_mesh = o3d.geometry.TriangleMesh.create_box(
            road_width, road_length, 0.01)
        plane_mesh.paint_uniform_color([0.2, 0.2, 0.2])  # Dark gray road
        plane_mesh.translate([-road_width/2, 0, -0.005])  # Center the road

        # Create lane marking lines
        grid_lines = []

        # Center line (double yellow)
        for offset in [-0.1, 0.1]:  # Two lines for center
            points = [[offset, 0, 0], [offset, road_length, 0]]
            lines = [[0, 1]]
            line_set = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(points),
                lines=o3d.utility.Vector2iVector(lines))
            line_set.paint_uniform_color([1, 1, 0])  # Yellow
            grid_lines.append(line_set)

        # Side lanes (white dashed)
        for x in [-3, 3]:  # Left and right lanes
            for y in range(0, road_length, 2):  # Dashed lines
                if y % 4 < 2:  # Create gaps
                    points = [[x, y, 0], [x, y + 2, 0]]
                    lines = [[0, 1]]
                    line_set = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector(points),
                        lines=o3d.utility.Vector2iVector(lines))
                    line_set.paint_uniform_color([1, 1, 1])  # White
                    grid_lines.append(line_set)

        self.ground_plane = {"mesh": plane_mesh, "grid": grid_lines}

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
        """Update visualization with speed optimizations"""
        if not self.use_3d_vis:
            return

        try:
            # Create a copy of the frame for visualization
            vis_frame = frame.copy()
            height, width = frame.shape[:2]

            # Draw detected lanes with transparency
            if lane_mask is not None:
                cv2.addWeighted(lane_mask, 0.4, vis_frame, 1, 0, vis_frame)

            # Draw driving corridor
            if hasattr(self, 'corridor_points'):
                # Simplified corridor visualization
                overlay = vis_frame.copy()
                cv2.fillPoly(overlay, self.corridor_points, (0, 255, 0))
                cv2.addWeighted(overlay, 0.2, vis_frame, 0.8, 0, vis_frame)

            # Draw objects with simplified 3D visualization
            for obj in vehicles + pedestrians + signs:
                bbox = obj['bbox']
                x1, y1, x2, y2 = [int(x) for x in bbox]
                color = obj['color']

                # Simple box with minimal 3D effect
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)

                # Distance estimation
                distance = 30 * (1 - y2 / height)

                # Simplified label
                label = f"{obj['type']}: {distance:.1f}m"
                cv2.putText(vis_frame, label, (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Simplified stats display
            cv2.putText(vis_frame, f"Vehicles: {len(vehicles)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 2)
            cv2.putText(vis_frame, f"People: {len(pedestrians)}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            cv2.putText(vis_frame, f"Signs: {len(signs)}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Show the visualization
            cv2.imshow("Autonomous Vehicle View", vis_frame)

        except Exception as e:
            print(f"\nWarning: Visualization update failed: {str(e)}")
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

                # Update SLAM visualization
                self.update_slam_visualization(points_3d, R, t)

        # Update previous frame data
        self.prev_frame = frame.copy()
        self.prev_keypoints = keypoints
        self.prev_descriptors = descriptors

        # Update visualization with all detections
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
