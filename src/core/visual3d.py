import open3d as o3d
import numpy as np
import threading
import queue
import time
from typing import List, Optional, Dict, Tuple
import torch
import multiprocessing
from concurrent.futures import ThreadPoolExecutor


class Visualizer3D:
    def __init__(self, use_gpu: bool = True):
        """Initialize 3D visualizer with GPU support if available"""
        # Initialize visualization in main thread for better window handling
        self._init_visualizer()

        # Multi-threading setup
        self.num_threads = max(multiprocessing.cpu_count() - 1, 1)
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_threads)

        # Point cloud and object data persistence
        self.accumulated_points = []
        self.accumulated_colors = []
        self.lane_points = []
        self.lane_colors = []
        self.objects_3d: Dict[int, Dict] = {}
        self.camera_trajectory = []
        self.current_view_params = None

        # Control flags
        self.pause_updates = False
        self.show_trajectory = True
        self.show_grid = True
        self.show_objects = True
        self.show_lanes = True

        # GPU support
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # Processing queues
        self.point_queue = queue.Queue()
        self.object_queue = queue.Queue()
        self.lane_queue = queue.Queue()
        self.running = True

        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_updates)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        print("\nVisualization Controls:")
        print("----------------------")
        print("Space: Pause/Resume visualization updates")
        print("R: Reset view")
        print("G: Toggle ground grid")
        print("T: Toggle camera trajectory")
        print("O: Toggle object bounding boxes")
        print("L: Toggle lane visualization")
        print("C: Clear all points")
        print("Mouse Controls:")
        print("- Left button + drag: Rotate")
        print("- Right button + drag: Pan")
        print("- Mouse wheel: Zoom")
        print("- Shift + Left button + drag: Roll")
        print("----------------------\n")

    def _init_visualizer(self):
        """Initialize Open3D visualizer"""
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name="3D Scene Understanding", width=1280, height=720, visible=True)

        # Register keyboard callbacks
        self.vis.register_key_callback(ord(" "), self._toggle_pause)
        self.vis.register_key_callback(ord("R"), self._reset_view)
        self.vis.register_key_callback(ord("G"), self._toggle_grid)
        self.vis.register_key_callback(ord("T"), self._toggle_trajectory)
        self.vis.register_key_callback(ord("O"), self._toggle_objects)
        self.vis.register_key_callback(ord("L"), self._toggle_lanes)
        self.vis.register_key_callback(ord("C"), self._clear_points)

        # Rendering options for more realistic look
        opt = self.vis.get_render_option()
        opt.point_size = 2.0
        opt.background_color = np.asarray([0, 0, 0])
        opt.light_on = True
        opt.point_show_normal = False
        opt.mesh_show_wireframe = True
        opt.mesh_show_back_face = True

        # View control
        self.view_control = self.vis.get_view_control()
        self._set_default_view()

        # Initialize geometries
        self.pcd = o3d.geometry.PointCloud()
        self.lane_pcd = o3d.geometry.PointCloud()
        self.vis.add_geometry(self.pcd)
        self.vis.add_geometry(self.lane_pcd)

        # Add coordinate frame
        self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.5)
        self.vis.add_geometry(self.coordinate_frame)

        # Add ground grid
        self.grid = self._create_ground_grid()
        self.vis.add_geometry(self.grid)

        # Force initial render
        self.vis.poll_events()
        self.vis.update_renderer()

    def _set_default_view(self):
        """Set default camera view parameters"""
        self.view_control.set_zoom(0.8)
        self.view_control.set_lookat([0, 0, 0])
        self.view_control.set_front([0, -0.5, -1])  # Angled view
        self.view_control.set_up([0, -1, 0.5])

    def _save_view(self):
        """Save current view parameters"""
        param = self.view_control.convert_to_pinhole_camera_parameters()
        self.current_view_params = param

    def _restore_view(self):
        """Restore saved view parameters"""
        if self.current_view_params is not None:
            self.view_control.convert_from_pinhole_camera_parameters(
                self.current_view_params)

    def _toggle_pause(self, vis):
        """Pause/Resume visualization updates"""
        self.pause_updates = not self.pause_updates
        print(
            f"Visualization updates {'paused' if self.pause_updates else 'resumed'}")
        return False

    def _reset_view(self, vis):
        """Reset camera view to default"""
        self._set_default_view()
        return False

    def _toggle_grid(self, vis):
        """Toggle ground grid visibility"""
        self.show_grid = not self.show_grid
        if self.show_grid:
            self.vis.add_geometry(self.grid)
        else:
            self.vis.remove_geometry(self.grid)
        return False

    def _toggle_trajectory(self, vis):
        """Toggle camera trajectory visibility"""
        self.show_trajectory = not self.show_trajectory
        self._update_trajectory()
        return False

    def _toggle_objects(self, vis):
        """Toggle object bounding boxes visibility"""
        self.show_objects = not self.show_objects
        self._update_objects()
        return False

    def _toggle_lanes(self, vis):
        """Toggle lane visualization"""
        self.show_lanes = not self.show_lanes
        if self.show_lanes:
            self.vis.add_geometry(self.lane_pcd)
        else:
            self.vis.remove_geometry(self.lane_pcd)
        return False

    def _clear_points(self, vis):
        """Clear all accumulated points"""
        self.accumulated_points = []
        self.accumulated_colors = []
        self.lane_points = []
        self.lane_colors = []
        self.pcd.points = o3d.utility.Vector3dVector(np.array([]))
        self.pcd.colors = o3d.utility.Vector3dVector(np.array([]))
        self.lane_pcd.points = o3d.utility.Vector3dVector(np.array([]))
        self.lane_pcd.colors = o3d.utility.Vector3dVector(np.array([]))
        self.vis.update_geometry(self.pcd)
        self.vis.update_geometry(self.lane_pcd)
        return False

    def _create_ground_grid(self):
        """Create a ground plane grid for better orientation"""
        grid_size = 20
        grid_resolution = 1.0
        points = []
        colors = []

        for i in range(-grid_size, grid_size + 1):
            for j in range(-grid_size, grid_size + 1):
                points.append([i * grid_resolution, j * grid_resolution, 0])
                if i == 0 or j == 0:
                    colors.append([0.5, 0.5, 0.5])  # Dimmer main axes
                else:
                    colors.append([0.2, 0.2, 0.2])  # Darker grid

        grid_pcd = o3d.geometry.PointCloud()
        grid_pcd.points = o3d.utility.Vector3dVector(np.array(points))
        grid_pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        return grid_pcd

    def create_bbox(self, center: np.ndarray, size: np.ndarray, R: np.ndarray = None) -> o3d.geometry.LineSet:
        """Create a bounding box line set"""
        if R is None:
            R = np.eye(3)

        # Define the 8 vertices of the box
        x, y, z = size / 2
        vertices = np.array([
            [-x, -y, -z], [x, -y, -z], [x, y, -z], [-x, y, -z],
            [-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]
        ])

        # Apply rotation and translation
        vertices = (R @ vertices.T).T + center

        # Define the 12 lines connecting vertices
        lines = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
            [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
            [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting edges
        ])

        # Create line set
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(vertices)
        line_set.lines = o3d.utility.Vector2iVector(lines)

        # Add orange color for better visibility
        line_set.colors = o3d.utility.Vector3dVector(
            np.array([[1, 0.7, 0] for _ in range(len(lines))]))
        return line_set

    def update_scene(self, points: np.ndarray, objects: List[Dict] = None, camera_pos: np.ndarray = None, lane_points: np.ndarray = None):
        """Update the entire 3D scene with points, objects, camera position, and lanes"""
        if not self.pause_updates:
            if points is not None:
                self.point_queue.put(('points', points))
            if objects is not None and self.show_objects:
                self.object_queue.put(('objects', objects))
            if camera_pos is not None and self.show_trajectory:
                self.camera_trajectory.append(camera_pos)
            if lane_points is not None and self.show_lanes:
                self.lane_queue.put(('lanes', lane_points))

    def _process_points(self, points: np.ndarray, is_lane: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Process point cloud data in parallel"""
        def process_chunk(chunk):
            mean = np.mean(chunk, axis=0)
            std = np.std(chunk, axis=0)
            mask = np.all(np.abs(chunk - mean) < 2.5 * std, axis=1)
            return chunk[mask]

        chunk_size = max(len(points) // self.num_threads, 1)
        chunks = [points[i:i + chunk_size]
                  for i in range(0, len(points), chunk_size)]
        processed_chunks = list(self.thread_pool.map(process_chunk, chunks))
        processed_points = np.vstack(processed_chunks)

        if is_lane:
            # Lane points are bright green
            colors = np.tile([0.0, 1.0, 0.0], (len(processed_points), 1))
        else:
            # Scene points use intensity-based coloring
            intensities = np.linalg.norm(processed_points, axis=1)
            colors = np.zeros((len(processed_points), 3))
            normalized_intensities = (
                intensities - np.min(intensities)) / (np.max(intensities) - np.min(intensities))

            # Create a more natural color gradient
            colors[:, 1] = 0.2 + 0.4 * normalized_intensities  # Green channel
            colors[:, 2] = 0.5 * normalized_intensities        # Blue channel
            colors[:, 0] = 0.3 * normalized_intensities        # Red channel

        return processed_points, colors

    def _update_trajectory(self):
        """Update camera trajectory visualization"""
        if len(self.camera_trajectory) > 1 and self.show_trajectory:
            lines = np.array(
                [[i, i+1] for i in range(len(self.camera_trajectory)-1)])
            # Yellow trajectory
            colors = np.array([[1, 1, 0] for _ in range(len(lines))])

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(
                np.array(self.camera_trajectory))
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)

            self.vis.add_geometry(line_set)

    def _update_objects(self):
        """Update object bounding boxes"""
        if self.show_objects:
            for obj_id, obj in self.objects_3d.items():
                if 'bbox' in obj:
                    bbox = self.create_bbox(obj['center'], obj['size'])
                    self.vis.add_geometry(bbox)

    def _process_updates(self):
        """Process updates in background thread"""
        while self.running:
            try:
                if not self.pause_updates:
                    self._save_view()

                    # Process point cloud updates
                    if not self.point_queue.empty():
                        _, points = self.point_queue.get_nowait()
                        if len(points) > 0:
                            processed_points, colors = self._process_points(
                                points, False)

                            if len(self.accumulated_points) == 0:
                                self.accumulated_points = processed_points
                                self.accumulated_colors = colors
                            else:
                                self.accumulated_points = np.vstack(
                                    [self.accumulated_points, processed_points])
                                self.accumulated_colors = np.vstack(
                                    [self.accumulated_colors, colors])

                            self.pcd.points = o3d.utility.Vector3dVector(
                                self.accumulated_points)
                            self.pcd.colors = o3d.utility.Vector3dVector(
                                self.accumulated_colors)
                            self.vis.update_geometry(self.pcd)

                    # Process lane updates
                    if not self.lane_queue.empty():
                        _, lane_points = self.lane_queue.get_nowait()
                        if len(lane_points) > 0:
                            processed_lanes, lane_colors = self._process_points(
                                lane_points, True)

                            if len(self.lane_points) == 0:
                                self.lane_points = processed_lanes
                                self.lane_colors = lane_colors
                            else:
                                self.lane_points = np.vstack(
                                    [self.lane_points, processed_lanes])
                                self.lane_colors = np.vstack(
                                    [self.lane_colors, lane_colors])

                            self.lane_pcd.points = o3d.utility.Vector3dVector(
                                self.lane_points)
                            self.lane_pcd.colors = o3d.utility.Vector3dVector(
                                self.lane_colors)
                            self.vis.update_geometry(self.lane_pcd)

                    # Process object updates
                    if not self.object_queue.empty():
                        _, objects = self.object_queue.get_nowait()
                        if self.show_objects:
                            for obj in objects:
                                if 'bbox' in obj:
                                    bbox = self.create_bbox(
                                        obj['center'], obj['size'])
                                    self.vis.add_geometry(bbox)

                    if self.show_trajectory:
                        self._update_trajectory()

                    self._restore_view()

                time.sleep(0.01)

            except queue.Empty:
                time.sleep(0.01)
                continue
            except Exception as e:
                print(f"Error in visualization thread: {str(e)}")
                continue

    def close(self):
        """Clean up resources"""
        self.running = False
        self.thread_pool.shutdown()
        self.vis.destroy_window()

    def update_visualization(self):
        """Update visualization - should be called from main thread"""
        if self.vis is not None and self.running:
            self.vis.poll_events()
            self.vis.update_renderer()
            return not self.vis.get_window_visible()
        return True
