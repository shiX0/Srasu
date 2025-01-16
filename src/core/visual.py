import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from plyfile import PlyData

# Function to load a PLY file and extract the vertices and their colors


def load_ply_with_colors(filename):
    ply_data = PlyData.read(filename)
    # Access the 'vertex' element which holds 3D point data
    vertex_data = ply_data['vertex']
    # Extract the x, y, z coordinates of vertices
    x = np.array(vertex_data['x'])
    y = np.array(vertex_data['y'])
    z = np.array(vertex_data['z'])

    # Extract the color data (assuming RGB color model is used)
    r = np.array(vertex_data['red'], dtype=float) / \
        255.0  # Normalize to [0, 1] range
    g = np.array(vertex_data['green'], dtype=float) / \
        255.0  # Normalize to [0, 1] range
    b = np.array(vertex_data['blue'], dtype=float) / \
        255.0  # Normalize to [0, 1] range

    # Combine the RGB values into a single color array
    colors = np.stack((r, g, b), axis=1)
    return x, y, z, colors

# Function to load trajectory data from .npy file


def load_trajectory(npy_file):
    # Load the trajectory data (assumes it is a 2D array with shape [n_points, 3])
    trajectory = np.load(npy_file)
    return trajectory

# Visualize the 3D points from the PLY file with colors and the trajectory


def visualize_ply_with_trajectory(ply_filename, trajectory_filename):
    # Load the PLY file data
    x, y, z, colors = load_ply_with_colors(ply_filename)

    # Load the trajectory data
    trajectory = load_trajectory(trajectory_filename)

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the point cloud with colors
    ax.scatter(x, y, z, c=colors, marker='.', label='Point Cloud', alpha=0.5)

    # Plot the trajectory as a line
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:,
            2], color='r', label='Trajectory', linewidth=2)

    # Set labels for axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the title
    ax.set_title('3D Point Cloud and Trajectory')

    # Show the plot
    ax.legend()
    plt.show()


# Replace with your PLY file and trajectory .npy file paths
ply_filename = 'slam_output/point_cloud_20250114_210328.ply'
trajectory_filename = 'slam_output/point_cloud_20250114_210328_trajectory.npy'


visualize_ply_with_trajectory(ply_filename, trajectory_filename)
