import argparse
import gc
import os
from os.path import isfile, join
import shutil
import sys
import time
from joblib import Parallel, delayed
import cv2
import numpy as np
from tqdm import tqdm

# sys.path
# sys.path.append(os.getcwd())
from utils.misc import save_img_to_file


PROCESSED_DATA_DIR = "../datasets/mmbreast/"


def parse_args():
    parser = argparse.ArgumentParser("Prepair mmbreast dataset.")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["rsna", "vindr", "miniddsm", "cmmd", "cddcesm", "bmcd"],
        help="Path to root directory of processed dataset.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of workers for cropping images.",
    )
    args = parser.parse_args()
    return args


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


def crop_save_single_func(src_path, save_path, threshold=0.3, save_backend=cv2):
    src_img = cv2.imread(src_path)
    assert src_img is not None, f"Failed to read image from {src_path}"
    # Convert to grayscale
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    # Apply GaussianBlur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply a binary threshold to create a binary image
    _, binary_mask = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY)
    # Get the crop boundaries based on the binary mask
    top, bottom, left, right = find_breast_region_edges(binary_mask, threshold)

    # Crop the original image based on the calculated edges
    cropped_image = src_img[top : bottom + 1, left : right + 1]
    save_img_to_file(save_path, cropped_image, backend=save_backend)


def crop_save(
    img_paths,
    save_paths,
    threshold,
    save_backend="cv2",
):
    assert len(img_paths) == len(save_paths)
    for i in tqdm(range(len(img_paths))):
        crop_save_single_func(img_paths[i], save_paths[i], threshold, save_backend)

    gc.collect()
    return


def crop_image_parallel(
    dcm_paths,
    save_paths,
    threshold=0.1,
    save_backend="cv2",
    parallel_n_jobs=2,
    parallel_n_chunks=4,
    joblib_backend="loky",
):
    assert len(dcm_paths) == len(save_paths)
    if parallel_n_jobs == 1:
        print("No parralel. Starting the tasks within current process.")
        return crop_save(
            dcm_paths,
            save_paths,
            threshold,
            save_backend,
        )
    else:
        num_samples = len(dcm_paths)
        num_samples_per_chunk = num_samples // parallel_n_chunks
        if num_samples % parallel_n_chunks > 0:
            num_samples_per_chunk += 1
        starts = [num_samples_per_chunk * i for i in range(parallel_n_chunks)]
        ends = [min(start + num_samples_per_chunk, num_samples) for start in starts]

        print(
            f"Starting {parallel_n_jobs} jobs with backend `{joblib_backend}`, {parallel_n_chunks} chunks..."
        )
        _ = Parallel(n_jobs=parallel_n_jobs, backend=joblib_backend)(
            delayed(crop_save)(
                dcm_paths[start:end],
                save_paths[start:end],
                threshold,
                save_backend,
            )
            for start, end in zip(starts, ends)
        )


def main(args):
    dataset = args.dataset
    cleaned_images_dir = join(PROCESSED_DATA_DIR, dataset, "cleaned_images")
    cropped_images_dir = join(PROCESSED_DATA_DIR, dataset, "cropped_images")
    if os.path.exists(cropped_images_dir):
        try:
            shutil.rmtree(cropped_images_dir)
        except OSError:
            os.remove(cropped_images_dir)
    os.makedirs(cropped_images_dir, exist_ok=True)
    list_fd = os.listdir(cleaned_images_dir)
    src_paths = [
        join(cleaned_images_dir, f)
        for f in list_fd
        if isfile(join(cleaned_images_dir, f))
    ]
    dst_paths = [
        join(cropped_images_dir, f)
        for f in list_fd
        if isfile(join(cleaned_images_dir, f))
    ]

    start = time.time()
    # CPU decode all others (exceptions) with dicomsdl
    crop_image_parallel(
        src_paths,
        dst_paths,
        save_backend="cv2",
        parallel_n_jobs=args.num_workers,
        parallel_n_chunks=args.num_workers,
        joblib_backend="loky",
    )
    end = time.time()
    gc.collect()
    print(f"Convert done in {end - start} sec")


if __name__ == "__main__":
    args = parse_args()
    main(args)
