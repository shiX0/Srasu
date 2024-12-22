from src.core.CammeraCalibration import calibrate_camera

if "__main__" == __name__:
    images_dir = 'calibration_images'
    chessboard_size = (9, 6)
    square_size = 0.025
    calibrate_camera('calibration_images', (9, 6), 0.025)