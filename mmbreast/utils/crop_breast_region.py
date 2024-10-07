import cv2
import numpy as np


def find_largest_true_sequence_indices(bool_list):
    max_sequence = 0  # To store the maximum length of True values
    current_sequence = 0  # To count the current sequence of True values
    start_index = -1  # To store the start index of the current sequence
    max_start = -1  # To store the start index of the largest sequence
    max_end = -1  # To store the end index of the largest sequence

    for i, value in enumerate(bool_list):
        if value:
            if current_sequence == 0:
                start_index = i  # Mark the start of the sequence
            current_sequence += 1
        else:
            if current_sequence > max_sequence:
                max_sequence = current_sequence
                max_start = start_index
                max_end = i - 1  # End index is the last True index
            current_sequence = 0  # Reset the current sequence

    # In case the list ends with a True sequence
    if current_sequence > max_sequence:
        max_start = start_index
        max_end = len(bool_list) - 1  # End index is the last True index

    return (max_start, max_end)


# Function to find crop boundaries based on white pixel percentage
def find_breast_region_edges(binary_mask, white_pixel_threshold=0.3):
    binary_mask[binary_mask == 255] = 1
    sum_rows = np.sum(binary_mask, axis=1)
    threshhold_row = np.mean(sum_rows) * white_pixel_threshold
    sum_cols = np.sum(binary_mask, axis=0)
    threshhold_col = np.mean(sum_cols) * white_pixel_threshold
    sum_rows[sum_rows < threshhold_row] = False
    sum_rows[sum_rows > threshhold_row] = True
    sum_cols[sum_cols < threshhold_col] = False
    sum_cols[sum_cols > threshhold_col] = True
    # Initialize edges
    top, bottom = find_largest_true_sequence_indices(sum_rows)
    left, right = find_largest_true_sequence_indices(sum_cols)
    return top, bottom, left, right


def crop_breast_region(src_img, threshold=0.3):
    assert src_img is not None
    # Convert to grayscale
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply a binary threshold to create a binary image
    _, binary_mask = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
    # Get the crop boundaries based on the binary mask
    top, bottom, left, right = find_breast_region_edges(binary_mask, threshold)

    # Crop the original image based on the calculated edges
    cropped_image = src_img[top : bottom + 1, left : right + 1]
    return cropped_image
