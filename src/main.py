from src.core.autonomous_perception import AutonomousPerception
import os
import sys
import cv2
import argparse

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


def verify_kitti_paths(kitti_base: str, date: str, drive: str) -> tuple:
    """Verify and construct KITTI paths"""
    # Construct paths
    date_dir = os.path.join(kitti_base, date)
    sequence_dir = os.path.join(date_dir, f"{date}_drive_{drive}_sync")
    image_dir = os.path.join(sequence_dir, "image_00", "data")
    calib_file = os.path.join(date_dir, "calib_cam_to_cam.txt")

    # Verify paths exist
    if not os.path.exists(date_dir):
        print(f"Error: Date directory not found: {date_dir}")
        print(f"Please make sure your KITTI path structure is correct:")
        print(f"kitti_dataset/")
        print(f"└── {date}/")
        print(f"    ├── calib_cam_to_cam.txt")
        print(f"    └── {date}_drive_{drive}_sync/")
        print(f"        └── image_00/")
        print(f"            └── data/")
        sys.exit(1)

    if not os.path.exists(sequence_dir):
        print(f"Error: Sequence directory not found: {sequence_dir}")
        sys.exit(1)

    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        sys.exit(1)

    if not os.path.exists(calib_file):
        print(f"Error: Calibration file not found: {calib_file}")
        print("Please download the calibration files from KITTI website")
        sys.exit(1)

    return date_dir, image_dir


def run_with_camera(camera_id: int = 0):
    """Run perception system with camera input"""
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)

    # Set camera parameters for better quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"Error: Could not open camera {camera_id}")
        sys.exit(1)

    # Initialize perception system
    perception = AutonomousPerception()

    print("Starting camera feed...")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press SPACE to pause/resume")
    print("- Press 's' to save current frame")

    paused = False
    frame_count = 0

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Process frame
            vis_frame = perception.process_frame(frame)
            frame_count += 1

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('s'):
            # Save current frame
            timestamp = cv2.getTickCount()
            filename = f"frame_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame to {filename}")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description='Run autonomous perception system')
    parser.add_argument('--use_camera', action='store_true',
                        help='Use camera input instead of KITTI dataset')
    parser.add_argument('--camera_id', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--kitti_path', type=str,
                        help='Path to KITTI dataset root (e.g., /path/to/kitti_dataset)')
    parser.add_argument('--date', type=str, default='2011_09_26',
                        help='Date of the sequence (e.g., 2011_09_26)')
    parser.add_argument('--drive', type=str, default='0009',
                        help='Drive number (e.g., 0009)')
    args = parser.parse_args()

    if args.use_camera:
        run_with_camera(args.camera_id)
    else:
        if not args.kitti_path:
            parser.error("--kitti_path is required when not using camera")

        # Verify and get paths
        date_dir, image_dir = verify_kitti_paths(
            args.kitti_path, args.date, args.drive)

        # Initialize perception system
        perception = AutonomousPerception(date_dir)

        # Process each image in sequence
        image_files = sorted(
            [f for f in os.listdir(image_dir) if f.endswith('.png')])
        if not image_files:
            print(f"No images found in {image_dir}")
            sys.exit(1)

        print(f"Found {len(image_files)} images to process")
        print("Starting processing...")
        print("Controls:")
        print("- Press 'q' to quit")
        print("- Press SPACE to pause/resume")
        print("- Press 'n' for next frame when paused")

        paused = False
        for img_name in image_files:
            # Read image
            img_path = os.path.join(image_dir, img_name)
            frame = cv2.imread(img_path)

            if frame is None:
                print(f"Failed to read image: {img_path}")
                continue

            # Process frame
            vis_frame = perception.process_frame(frame)

            # Handle keyboard input
            while True:
                key = cv2.waitKey(1 if not paused else 0) & 0xFF
                if key == ord('q'):
                    cv2.destroyAllWindows()
                    sys.exit(0)
                elif key == ord(' '):
                    paused = not paused
                    print("Paused" if paused else "Resumed")
                elif key == ord('n') and paused:
                    break
                elif not paused:
                    break

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
