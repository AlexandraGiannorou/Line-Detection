import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_image(title, image, cmap=None):
    """Helper function to display an image using matplotlib."""
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def pipeline(image):
    height = image.shape[0]
    width = image.shape[1]

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    show_image('Gray Image', gray_image, cmap='gray')

    # Apply Canny edge detection
    cannyed_image = cv2.Canny(image, 100, 200)
    show_image('Canny Edge Detection', cannyed_image, cmap='gray')

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(cannyed_image,
                            rho=1,
                            theta=np.pi/180,
                            threshold=80,
                            minLineLength=100,
                            maxLineGap=10)

    if lines is not None:
        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0
                if abs(slope) < 0.1:  # Ignore near-horizontal lines
                    continue
                if slope < 0:  # Left line
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                elif slope > 0:  # Right line
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])

        # Draw the detected lines
        line_img = np.zeros((height, width, 3), dtype=np.uint8)
        if left_line_x and left_line_y:
            left_fit = np.polyfit(left_line_y, left_line_x, 1)
            left_y1 = height
            left_y2 = int(height * 0.6)
            left_x1 = int(left_fit[0] * left_y1 + left_fit[1])
            left_x2 = int(left_fit[0] * left_y2 + left_fit[1])
            cv2.line(line_img, (left_x1, left_y1), (left_x2, left_y2), (255, 0, 0), 5)

        if right_line_x and right_line_y:
            right_fit = np.polyfit(right_line_y, right_line_x, 1)
            right_y1 = height
            right_y2 = int(height * 0.6)
            right_x1 = int(right_fit[0] * right_y1 + right_fit[1])
            right_x2 = int(right_fit[0] * right_y2 + right_fit[1])
            cv2.line(line_img, (right_x1, right_y1), (right_x2, right_y2), (0, 255, 0), 5)

        # Combine the original image with the line image
        result_image = cv2.addWeighted(image, 0.8, line_img, 1, 0)
        show_image('Result Image', cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

# Load the image
image = cv2.imread("C:/Users/alexia/Desktop/py4e/ntua.jpg")
if image is None:
    print("Error: Image not found or unable to load.")
    exit()

# Process the image
pipeline(image)