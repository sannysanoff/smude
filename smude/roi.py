__author__ = "Simon Waloschek"

import logging
from typing import Tuple

import cv2
import numpy as np
from scipy import ndimage
from skimage.exposure import equalize_adapthist
from skimage.morphology import footprint_rectangle
from skimage.segmentation import felzenszwalb


def get_border(image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Get border widths of binary image for all 4 edges.

    Parameters
    ----------
    image : np.ndarray
        Binary image.

    Returns
    -------
    x_start : int
        X-Coordinate of first non-white pixel.
    x_end : int
        X-Coordinate of last non-white pixel.
    y_start : int
        Y-Coordinate of first non-white pixel.
    y_end : int
        Y-Coordinate of last non-white pixel.
    """
    cols, rows = np.where(image == 0)
    x_start = np.min(cols)
    x_end = np.max(cols) + 1

    y_start = np.min(rows)
    y_end = np.max(rows) + 1

    return x_start, x_end, y_start, y_end


def auto_canny(image, sigma=0):
    v = np.median(image)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(90, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    return edged


def extract_roi_mask(image: np.ndarray, min_hull_ratio: float = 0.4, verbose: bool = False) -> Tuple[np.ndarray, float]:
    """
    Extract region of interest (ROI) for the given image.

    Parameters
    ----------
    image : np.ndarray
        Input document image covering the entire ROI.
    min_hull_ratio : float, optional
        Minimum ratio All/ROI for counting as "success". 
        Range: 0.1-0.9, by default 0.4.
    verbose : bool, optional
        Enable verbose logging, by default False.

    Returns
    -------
    mask_fullsize : np.ndarray
        Binary image respresenting the ROI. White pixels (1) = ROI.
    mask_ratio : float
        Pixel ratio (width * height) / ROI.

    Raises
    ------
    Exception
        If the minimum desired ratio could not be achieved, an error is raised.
    """
    logging.info('Starting ROI extraction')
    # Scale image to fixed size
    size = 512
    width, height, _ = image.shape
    image_resized = cv2.resize(image, (size, size))

    if verbose:
        logging.info(f'Original image size: {width}x{height}, resized to {size}x{size}')

    logging.info('Preprocessing image for segmentation')
    image_gray = cv2.cvtColor(image_resized, cv2.COLOR_RGB2GRAY)
    image_eq = equalize_adapthist(image_gray) * 255
    image_canny = auto_canny(image_eq.astype(np.uint8))
    image_canny = cv2.morphologyEx(
        image_canny, cv2.MORPH_DILATE, kernel=footprint_rectangle((2, 2))
    )
    
    logging.info('Performing image segmentation')
    image_segmented = felzenszwalb(image_canny, scale=1000, sigma=0.3, min_size=50)

    segment_sizes = np.bincount(image_segmented.flatten())
    segments = np.argsort(-segment_sizes)
    
    if verbose:
        logging.info(f'Segmentation found {len(segments)} segments, largest has {segment_sizes[segments[0]]} pixels')

    # Iterate over 5 largest segments, starting from largest
    for s in segments[:5]:
        # Get segment and fill all holes
        segment = image_segmented == s
        hull = ndimage.binary_fill_holes(segment)

        # Removes areas that are only connected by few pixels to the hull
        hull_opened = cv2.morphologyEx(
            hull.astype(np.uint8), cv2.MORPH_OPEN, kernel=footprint_rectangle((20, 20))
        )

        # Take center blob
        #blobs_segmented = measure.label(hull_opened)
        _, blobs_segmented = cv2.connectedComponents(hull_opened, connectivity=4)
        center_blob_label = blobs_segmented[size // 2, size // 2]
        hull = blobs_segmented == center_blob_label

        # Exit if hull_ratio is sufficient
        hull_ratio = np.sum(hull) / (size**2)
        border_touches = int(np.any(hull[0])) + int(np.any(hull[size-1])) + int(np.any(hull[:,0])) + int(np.any(hull[:,size-1]))
        
        if verbose:
            logging.info(f'Segment {s}: hull_ratio={hull_ratio:.4f}, border_touches={border_touches}')
        
        if hull_ratio >= min_hull_ratio and border_touches < 4:
            if verbose:
                logging.info(f'Selected segment {s} with hull_ratio={hull_ratio:.4f}')
            break

    # Raise error if hull_ratio criterion could not be met
    if hull_ratio < min_hull_ratio:
        raise Exception('ROI could not be computed')

    logging.info('Finalizing ROI mask')
    # Resize mask back to original image size
    mask_fullsize = cv2.resize(hull.astype(np.uint8), (height, width))

    # Remove outer pixels so that dark residual pixels are removed
    mask_fullsize = cv2.morphologyEx(
        mask_fullsize, cv2.MORPH_ERODE, kernel=footprint_rectangle((25, 25))
    )
    mask_ratio = np.sum(hull) / (size**2)

    logging.info(f'ROI extraction completed with mask ratio: {mask_ratio:.4f}')
    return mask_fullsize, mask_ratio
