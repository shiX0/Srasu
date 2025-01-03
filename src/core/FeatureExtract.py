import os
import cv2 as cv
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.abspath(os.path.join(script_dir, '../test/preview.png'))

# Load the image in grayscale
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: Could not load the image from {image_path}")
    exit()
else:
    print("Image loaded successfully")

# Initiate ORB detector
orb = cv.ORB_create()

# Detect keypoints and compute descriptors
kp, des = orb.detectAndCompute(img, None)



# Draw keypoints
img2 = cv.drawKeypoints(img, kp, None, color=(255, 0, 0), flags=cv.DrawMatchesFlags_DEFAULT)

# Print ORB parameters and keypoint count
print("Total Keypoints detected by ORB: {}".format(len(kp)))
print("Descriptor shape: {}".format(des.shape))

# Show ORB keypoints
plt.imshow(img2, cmap='gray')
plt.title('ORB Keypoints')
plt.show()
