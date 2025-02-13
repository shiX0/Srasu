# SRASU (Spatial Reconstruction and Scene Understanding)

A comprehensive repository for autonomous vehicle perception using SLAM (Simultaneous Localization and Mapping), computer vision, and 3D reconstruction techniques.

## Overview

SRASU implementations for spatial reconstruction and scene understanding, specifically designed for autonomous vehicles and robots. The project utilizes various libraries including:

- OpenCV for image processing
- Open3D for 3D data manipulation
- SLAM algorithms for mapping and localization

## Key Features

- Real-time SLAM implementation
- 3D scene reconstruction
- Object detection and tracking
- Spatial mapping
- Scene understanding algorithms

## Dependencies

- OpenCV
- Open3D
- Python 3.x
- NumPy
- Additional SLAM libraries

## Team Members

- Aashista Karki
- Shishir Sharma

## Contact

For questions and contributions, please open an issue in the repository.

Command to run the code:

for dataset:
python -m src.main --kitti_path "C:\Users\nisch\OneDrive\Desktop\Srasu\Srasu\kitti_dataset"

for live video:
python -m src.main --use_camera

python -m src.main --kitti_path "C:\Users\nisch\OneDrive\Desktop\Srasu\Srasu\kitti_dataset"


# Run depth estimation with camera (default mode is depth visualization)
python -m src.depth_main --use_camera

# Run with 3D reconstruction mode
python -m src.depth_main --use_camera --mode reconstruction


# Run depth estimation on KITTI dataset (default mode is depth visualization)
python -m src.depth_main --kitti_path "path/to/kitti_dataset"

# Run with 3D reconstruction mode
python -m src.depth_main --kitti_path "path/to/kitti_dataset" --mode reconstruction
