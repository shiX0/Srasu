import os
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt


# Get the absolute path of the image relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.abspath(os.path.join(script_dir, '../test/preview.png'))

# Load the image in grayscale
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
if img is None:
    print(f"Error: Could not load the image from {image_path}")
    exit()
else:
    print("Image loaded successfully")

# Initiate FAST object with default values
fast = cv.FastFeatureDetector_create()
kp = fast.detect(img, None)

# Draw keypoints (without displaying yet)
img_with_kp = cv.drawKeypoints(img, kp, None, color=(255, 0, 0))

# Plot the image with keypoints
fig, ax = plt.subplots()
ax.imshow(img_with_kp, cmap='gray')
ax.set_title("Hover over keypoints to see details")

# Convert keypoints to a list of (x, y) coordinates
keypoint_coords = [(int(k.pt[0]), int(k.pt[1])) for k in kp]

# Create an annotation for displaying keypoint details
annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                    bbox=dict(boxstyle="round", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
annot.set_visible(False)


def update_annot(ind):
    """Update the annotation with keypoint details."""
    x, y = keypoint_coords[ind["ind"][0]]
    annot.xy = (x, y)
    text = f"X: {x}, Y: {y}"
    annot.set_text(text)
    annot.get_bbox_patch().set_alpha(0.8)


def hover(event):
    """Event handler for mouse hover."""
    vis = annot.get_visible()
    if event.inaxes == ax:
        for i, (x, y) in enumerate(keypoint_coords):
            if abs(event.xdata - x) < 5 and abs(event.ydata - y) < 5:  # Check proximity
                update_annot({"ind": [i]})
                annot.set_visible(True)
                fig.canvas.draw_idle()
                return
        if vis:
            annot.set_visible(False)
            fig.canvas.draw_idle()


# Connect the hover event to the figure
fig.canvas.mpl_connect("motion_notify_event", hover)

plt.show()
