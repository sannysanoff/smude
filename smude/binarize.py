__author__ = "Simon Waloschek"

import logging
import numpy as np
import cv2
from skimage.morphology import remove_small_holes
from skimage.segmentation import flood_fill

#@profile
def binarize(image: np.ndarray, holes_threshold: float = 20) -> np.ndarray:
    """
    Binarize image using Sauvola algorithm.

    Parameters
    ----------
    image : np.ndarray
        RGB image to binarize.
    holes_threshold : float, optional
        Pixel areas covering less than the given number of pixels are removed in the process, by default 20.

    Returns
    -------
    binarized : np.ndarray
        Binarized and filtered image.
    """

    logging.info('Starting binarization process')
    # Extract brightness channel from HSV-converted image
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(256, 256))
    image_eq = clahe.apply(image_gray)

    # Threshold using Sauvola algorithm
    logging.info('Applying Sauvola threshold')
    binary_sauvola = cv2.ximgproc.niBlackThreshold(image_eq, 255, k=0.25, blockSize=51, type=cv2.THRESH_BINARY, binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)

    # Remove small objects
    #binary_cleaned = 1.0 * remove_small_holes(binary_sauvola, area_threshold=holes_threshold)

    # Remove thick black border (introduced during thresholding)
    logging.info('Removing borders using flood fill')
    binary_sauvola = flood_fill(binary_sauvola, (0, 0), 0)
    binary_sauvola = flood_fill(binary_sauvola, (0, 0), 1)

    logging.info('Binarization completed')
    return binary_sauvola.astype(bool)
