import numpy as np
import cv2
import matplotlib.pyplot as plt


# Function to calculate LTCP descriptor
def calculate_ltcp(image, triangles):
    """
    Compute the Local Triangular Coded Pattern (LTCP) for a given image.

    Parameters:
    - image: 2D array (grayscale image)
    - triangles: List of triangular pixel indices (relative to a pixel's neighborhood)

    Returns:
    - ltcp_image: 2D array of LTCP values
    """
    rows, cols = image.shape
    ltcp_image = np.zeros((rows, cols), dtype=np.uint8)

    # Define the 3x3 neighborhood for triangular patterns
    neighborhood = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 0), (0, 1),
                    (1, -1), (1, 0), (1, 1)]

    # Iterate over every pixel in the image
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            pixel_patterns = []

            # Extract 3x3 neighborhood
            neighbors = [(i + dx, j + dy) for dx, dy in neighborhood]
            pixel_values = [image[nx, ny] for nx, ny in neighbors]

            # Loop over each triangle configuration
            for triangle in triangles:
                p1, p2, p3 = triangle
                value = 0

                # Compare intensities in the triangle
                if pixel_values[p1] > pixel_values[p2]:
                    value |= 1
                if pixel_values[p2] > pixel_values[p3]:
                    value |= 2
                if pixel_values[p1] > pixel_values[p3]:
                    value |= 4

                pixel_patterns.append(value)

            # Encode the pixel's pattern as an LTCP value (e.g., by averaging or selecting key patterns)
            ltcp_image[i, j] = np.mean(pixel_patterns)

    return ltcp_image


# Example usage
# Read a grayscale image
image = cv2.imread('new4.jpg', cv2.IMREAD_GRAYSCALE)

image = cv2.resize(image, (224, 224))

# Define triangular neighborhood patterns (3 points forming triangles)
triangles = [(0, 1, 3), (1, 2, 5), (3, 4, 6), (4, 5, 8)]

# Calculate LTCP descriptor
ltcp_image = calculate_ltcp(image, triangles)




# https://github.com/anita-hu/MSAF/blob/master/MSAF.py



# Here’s a step-by-step explanation of what we did in the above code:
#
# Image Input: We start by reading a grayscale image, where each pixel's intensity is represented by a value (e.g., from 0 to 255).
#
# Neighborhood Definition: For each pixel in the image, we define a 3x3 grid of neighboring pixels around it. This 3x3 region forms the local area where triangular patterns are considered.
#
# Triangle Formation: We create multiple triangular configurations from the 3x3 neighborhood. Each triangle is made of three specific pixels, and we consider various combinations of these pixels to form different triangles.
#
# Pixel Comparisons: For each triangle, we compare the intensity (brightness) values of the three pixels involved in the triangle:
#
# Compare the first pixel with the second.
# Compare the second pixel with the third.
# Compare the first pixel with the third.
# Binary Pattern Encoding: Based on the comparison results:
#
# If the first pixel's intensity is greater than the second pixel’s, the least significant bit (LSB) is set to 1.
# If the second pixel’s intensity is greater than the third’s, the second bit is set to 1.
# If the first pixel’s intensity is greater than the third’s, the most significant bit is set to 1.
# Bitwise OR Assignment: The bitwise OR operator (|=) is used to combine the results of these comparisons, forming a 3-bit binary value (ranging from 0 to 7) that encodes the intensity relationships in the triangle.
#
# LTCP Value Calculation: For each pixel, we compute binary patterns for multiple triangles within the neighborhood. These patterns describe the local texture around the pixel. The final LTCP value for the pixel can be derived by averaging or summarizing these binary patterns.
#
# Output Image Creation: The LTCP values are stored in a new image, where each pixel's LTCP value represents the texture in that local region of the original image.
#
# Feature Extraction: The LTCP image captures the local intensity variations in the form of binary patterns. This transformed image can then be used as input to a CNN for tasks like disease detection, enhancing texture-based features.