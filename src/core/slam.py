import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging
import datetime
import os
from .visual3d import Visualizer3D


class ORBSLAM:
    def __init__(self, camera_index: int = 0, output_dir: str = "slam_output"):
        """
        Initialize the ORB-SLAM system

        Args:
            camera_index (int): Index of the camera device
            output_dir (str): Directory to save output files
        """
        self.camera_index = camera_index
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Initialize 3D visualizer first
        self.visualizer = Visualizer3D(use_gpu=True)

        # Initialize logging
        self.setup_logging()

        # ORB Feature Detector with optimized parameters
        self.orb = cv2.ORB_create(
            nfeatures=10000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=31,
            firstLevel=0,
            WTA_K=2,
            patchSize=31,
            fastThreshold=20
        )

        # Feature Matcher
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Video capture
        self.cap = None

        # Previous frame data
        self.prev_frame = None
        self.prev_keypoints = None
        self.prev_descriptors = None

        # 3D reconstruction data
        self.point_cloud = []
        self.camera_positions = []
        self.frame_poses = []

        # Camera parameters (should be calibrated)
        self.K = np.array([[718.856, 0, 607.1928],
                          [0, 718.856, 185.2157],
                          [0, 0, 1]], dtype=np.float32)  # KITTI camera parameters
        self.dist_coeffs = np.zeros((4, 1))

        # Track quality metrics
        self.min_matches = 10
        self.min_tracking_quality = 0.85
        self.last_tracking_quality = 1.0

        # Frame counter
        self.frame_count = 0

        self.logger.info("SLAM system initialized successfully")

    def setup_logging(self):
        """Configure logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.output_dir, 'slam.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def start_video_capture(self):
        """Initialize video capture"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                raise RuntimeError("Failed to open camera")

            # Set camera parameters
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)

            self.logger.info("Camera initialized successfully")
        except Exception as e:
            self.logger.error(f"Error initializing camera: {str(e)}")
            raise

    def stop_video_capture(self):
        """Release video capture and close windows"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.visualizer.close()
        self.logger.info("Camera released and windows closed")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better feature detection

        Args:
            frame (np.ndarray): Input frame

        Returns:
            np.ndarray: Preprocessed frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

        return blurred

    def feature_extraction(self, frame: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Extract ORB features from frame

        Args:
            frame (np.ndarray): Input frame

        Returns:
            Tuple[List, np.ndarray]: Keypoints and descriptors
        """
        try:
            keypoints, descriptors = self.orb.detectAndCompute(frame, None)
            if keypoints is None or descriptors is None:
                raise ValueError("No features detected")
            return keypoints, descriptors
        except Exception as e:
            self.logger.warning(f"Feature extraction failed: {str(e)}")
            return [], None

    def feature_matching(self, descriptors1: np.ndarray, descriptors2: np.ndarray) -> List:
        """
        Match features between frames

        Args:
            descriptors1 (np.ndarray): Descriptors from first frame
            descriptors2 (np.ndarray): Descriptors from second frame

        Returns:
            List: Filtered matches
        """
        try:
            matches = self.bf.match(descriptors1, descriptors2)

            # Filter matches based on distance
            if len(matches) > 0:
                min_dist = min(matches, key=lambda x: x.distance).distance
                good_matches = [
                    m for m in matches if m.distance < max(2.5 * min_dist, 30.0)]
                return good_matches
            return []
        except Exception as e:
            self.logger.warning(f"Feature matching failed: {str(e)}")
            return []

    def process_frame(self, frame: np.ndarray):
        """
        Process a single frame

        Args:
            frame (np.ndarray): Input frame
        """
        try:
            # Preprocess frame
            processed_frame = self.preprocess_frame(frame)

            # Extract features
            keypoints, descriptors = self.feature_extraction(processed_frame)

            if self.prev_descriptors is not None and descriptors is not None:
                # Match features
                matches = self.feature_matching(
                    self.prev_descriptors, descriptors)

                if len(matches) > self.min_matches:
                    # Extract point cloud and camera pose
                    R, t = self.extract_point_cloud(
                        matches, self.prev_keypoints, keypoints)

                    # Update camera position
                    if len(self.camera_positions) > 0:
                        last_pos = self.camera_positions[-1]
                        new_pos = last_pos + R @ t.ravel()
                    else:
                        new_pos = np.zeros(3)
                    self.camera_positions.append(new_pos)

                    # Get detected objects from the frame
                    objects_3d = []
                    if hasattr(frame, 'detected_objects'):
                        for obj in frame.detected_objects:
                            # Convert 2D bbox to 3D
                            center_3d = np.array([
                                obj['bbox'][0] + obj['bbox'][2]/2,
                                obj['bbox'][1] + obj['bbox'][3]/2,
                                0
                            ])
                            size_3d = np.array([
                                obj['bbox'][2],
                                obj['bbox'][3],
                                2.0
                            ])
                            objects_3d.append({
                                'center': center_3d,
                                'size': size_3d,
                                'type': obj.get('type', 'unknown')
                            })

                    # Detect lanes
                    lane_mask = self.detect_lanes(frame)
                    lane_points = []
                    if lane_mask is not None:
                        # Convert lane mask to 3D points
                        lane_y, lane_x = np.where(lane_mask > 0)
                        if len(lane_x) > 0:
                            # Project lane points to 3D
                            lane_points_2d = np.column_stack([lane_x, lane_y])
                            lane_points_norm = cv2.undistortPoints(
                                lane_points_2d.reshape(-1, 1, 2),
                                self.K,
                                self.dist_coeffs
                            ).reshape(-1, 2)

                            # Assume lanes are on the ground plane (z=0)
                            lane_points = np.column_stack([
                                lane_points_norm * 5.0,  # Scale for better visibility
                                np.zeros(len(lane_points_norm))
                            ])

                    # Update point cloud visualization if we have enough points
                    if len(self.point_cloud) > 100:
                        points = np.array(self.point_cloud)
                        # Update the entire 3D scene
                        self.visualizer.update_scene(
                            points=points,
                            objects=objects_3d,
                            camera_pos=new_pos,
                            lane_points=lane_points if len(
                                lane_points) > 0 else None
                        )

                    # Draw matches on frame for visualization
                    vis_frame = cv2.drawMatches(
                        self.prev_frame, self.prev_keypoints,
                        processed_frame, keypoints,
                        matches[:100], None,
                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
                    )
                    cv2.imshow('Feature Matching', vis_frame)

            # Update previous frame data
            self.prev_frame = processed_frame
            self.prev_keypoints = keypoints
            self.prev_descriptors = descriptors
            self.frame_count += 1

        except Exception as e:
            self.logger.error(f"Frame processing failed: {str(e)}")

    def detect_lanes(self, frame: np.ndarray) -> np.ndarray:
        """Detect lanes in the frame"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply Canny edge detection
            edges = cv2.Canny(blurred, 50, 150)

            # Create ROI mask
            height, width = edges.shape
            mask = np.zeros_like(edges)
            roi_vertices = np.array([
                [(0, height),
                 (width * 0.45, height * 0.6),
                 (width * 0.55, height * 0.6),
                 (width, height)]
            ], dtype=np.int32)
            cv2.fillPoly(mask, roi_vertices, 255)
            masked_edges = cv2.bitwise_and(edges, mask)

            # Detect lines using HoughLinesP
            lines = cv2.HoughLinesP(
                masked_edges,
                rho=1,
                theta=np.pi/180,
                threshold=50,
                minLineLength=height * 0.3,
                maxLineGap=height * 0.05
            )

            # Create lane mask
            lane_mask = np.zeros_like(frame)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(lane_mask, (x1, y1), (x2, y2), (255, 255, 255), 2)

            return lane_mask[:, :, 0]  # Return single channel

        except Exception as e:
            self.logger.error(f"Lane detection failed: {str(e)}")
            return None

    def filter_point_cloud(self, points: np.ndarray) -> np.ndarray:
        """
        Filter point cloud to remove outliers

        Args:
            points (np.ndarray): Input points

        Returns:
            np.ndarray: Filtered points
        """
        try:
            # Remove statistical outliers
            mean = np.mean(points, axis=0)
            std = np.std(points, axis=0)
            filtered_points = points[np.all(
                np.abs(points - mean) < 3 * std, axis=1)]

            return filtered_points
        except Exception as e:
            self.logger.error(f"Point cloud filtering failed: {str(e)}")
            return points

    def save_point_cloud(self, base_filename: str = "point_cloud"):
        """
        Save point cloud in multiple formats

        Args:
            base_filename (str): Base filename for saving
        """
        try:
            if not self.point_cloud:
                self.logger.warning("No points to save")
                return

            points = np.array(self.point_cloud)
            filtered_points = self.filter_point_cloud(points)

            # Create timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            base_path = os.path.join(
                self.output_dir, f"{base_filename}_{timestamp}")

            # Save as PLY
            self.save_as_ply(filtered_points, f"{base_path}.ply")

            # Save as NumPy array
            np.save(f"{base_path}.npy", filtered_points)

            # Save camera trajectory
            np.save(f"{base_path}_trajectory.npy",
                    np.array(self.camera_positions))

            self.logger.info(
                f"Saved {len(filtered_points)} points to {base_path}")

        except Exception as e:
            self.logger.error(f"Failed to save point cloud: {str(e)}")

    def save_as_ply(self, points: np.ndarray, filename: str):
        """
        Save point cloud as PLY file

        Args:
            points (np.ndarray): Points to save
            filename (str): Output filename
        """
        try:
            # Generate random colors for each point (you could use actual colors if available)
            colors = np.random.randint(0, 255, (len(points), 3))
            with open(filename, 'w') as f:
                f.write("ply\n")
                f.write("format ascii 1.0\n")
                f.write(f"element vertex {len(points)}\n")
                f.write("property float x\n")
                f.write("property float y\n")
                f.write("property float z\n")
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
                f.write("end_header\n")
                for point, color in zip(points, colors):
                    f.write(
                        f"{point[0]} {point[1]} {point[2]} {color[0]} {color[1]} {color[2]}\n")
            self.logger.info(f"Point cloud saved as PLY to {filename}")
        except Exception as e:
            self.logger.error(f"Failed to save PLY file: {str(e)}")

    def extract_point_cloud(self, matches: List, keypoints1: List, keypoints2: List) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract 3D points from matched features with improved accuracy

        Args:
            matches (List): Feature matches
            keypoints1 (List): Keypoints from first frame
            keypoints2 (List): Keypoints from second frame

        Returns:
            Tuple[np.ndarray, np.ndarray]: Rotation matrix and translation vector
        """
        try:
            if len(matches) < 8:
                return np.eye(3), np.zeros((3, 1))

            # Extract matched points
            points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
            points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])

            # Normalize points
            points1_norm = cv2.undistortPoints(
                points1.reshape(-1, 1, 2), self.K, self.dist_coeffs)
            points2_norm = cv2.undistortPoints(
                points2.reshape(-1, 1, 2), self.K, self.dist_coeffs)

            # Find essential matrix with stricter RANSAC parameters
            E, mask = cv2.findEssentialMat(
                points1_norm, points2_norm,
                focal=1.0, pp=(0., 0.),
                method=cv2.RANSAC,
                prob=0.999,
                threshold=0.0003  # Stricter threshold
            )

            if E is None or E.shape != (3, 3):
                return np.eye(3), np.zeros((3, 1))

            # Recover pose with more accurate parameters
            _, R, t, mask = cv2.recoverPose(
                E, points1_norm, points2_norm, mask=mask)

            # Triangulate points with better accuracy
            proj1 = np.hstack((np.eye(3), np.zeros((3, 1))))
            proj2 = np.hstack((R, t))

            # Scale matrices appropriately
            proj1 = self.K @ proj1
            proj2 = self.K @ proj2

            points4D = cv2.triangulatePoints(proj1, proj2, points1, points2)
            points3D = points4D[:3] / points4D[3]

            # Filter points based on reprojection error
            max_error = 2.0  # pixels
            valid_points = []
            for i, point3D in enumerate(points3D.T):
                if mask[i]:
                    # Project point back to both images
                    proj1_point = proj1 @ np.append(point3D, 1)
                    proj2_point = proj2 @ np.append(point3D, 1)

                    proj1_point = proj1_point[:2] / proj1_point[2]
                    proj2_point = proj2_point[:2] / proj2_point[2]

                    error1 = np.linalg.norm(proj1_point - points1[i])
                    error2 = np.linalg.norm(proj2_point - points2[i])

                    if error1 < max_error and error2 < max_error:
                        valid_points.append(point3D)

            # Store filtered 3D points
            if len(valid_points) > 0:
                self.point_cloud.extend(valid_points)

            return R, t

        except Exception as e:
            self.logger.error(f"Point cloud extraction failed: {str(e)}")
            return np.eye(3), np.zeros((3, 1))

    def run(self):
        """Run the SLAM system"""
        self.logger.info("Starting SLAM system")
        self.start_video_capture()

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame")
                    break

                self.process_frame(frame)

                # Display tracking quality
                cv2.putText(frame, f"Quality: {self.last_tracking_quality:.2f}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Frame', frame)

                # Update 3D visualization
                if self.visualizer.update_visualization():
                    break  # Break if visualization window is closed

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

                # Auto-save periodically
                if self.frame_count % 300 == 0:  # Save every 300 frames
                    self.save_point_cloud()

        except KeyboardInterrupt:
            self.logger.info("SLAM system stopped by user")
        except Exception as e:
            self.logger.error(f"SLAM system error: {str(e)}")
        finally:
            self.save_point_cloud()
            self.stop_video_capture()
            self.logger.info("SLAM system shutdown complete")


if __name__ == "__main__":
    slam = ORBSLAM()
    slam.run()
