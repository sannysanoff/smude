__author__ = "Simon Waloschek"

import logging
import numpy as np
import cv2
from skimage.morphology import remove_small_holes
from skimage.segmentation import flood_fill

#@profile
def binarize(image: np.ndarray, holes_threshold: float = 20, noise_reduction: dict = None, verbose: bool = False, threshold: int = 128, sauvola_k: float = 0.25) -> np.ndarray:
    """
    Binarize image using Sauvola algorithm.

    Parameters
    ----------
    image : np.ndarray
        RGB image to binarize.
    holes_threshold : float, optional
        Pixel areas covering less than the given number of pixels are removed in the process.
        Range: 1-100, by default 20.
    noise_reduction : dict, optional
        Dictionary controlling noise reduction aggressiveness. Keys:
        - 'hole_removal': float, multiplier for hole removal threshold (range: 0.0-5.0, default 1.0)
        - 'opening_strength': float, kernel size multiplier for opening operation (range: 0.0-5.0, default 1.0)  
        - 'closing_strength': float, kernel size multiplier for closing operation (range: 0.0-5.0, default 1.0)
        - 'median_strength': float, kernel size multiplier for median filtering (range: 0.0-5.0, default 1.0)
        Set to None or empty dict to disable noise reduction.

    Returns
    -------
    binarized : np.ndarray
        Binarized and filtered image.
    """

    logging.info('Starting binarization process')
    logging.info(f'Using threshold: {threshold}')
    # Extract brightness channel from HSV-converted image
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=0.01, tileGridSize=(256, 256))
    image_eq = clahe.apply(image_gray)

    # Threshold using Sauvola algorithm
    logging.info('Applying Sauvola threshold')
    binary_sauvola = cv2.ximgproc.niBlackThreshold(image_eq, 255, k=sauvola_k, blockSize=51, type=cv2.THRESH_BINARY, binarizationMethod=cv2.ximgproc.BINARIZATION_SAUVOLA)
    
    # Save intermediate image after Sauvola threshold
    if verbose:
        cv2.imwrite('verbose_sauvola_threshold.jpg', binary_sauvola)
        logging.info('Saved verbose image: verbose_sauvola_threshold.jpg')

    # Remove small objects
    #binary_cleaned = 1.0 * remove_small_holes(binary_sauvola, area_threshold=holes_threshold)

    # Remove thick black border (introduced during thresholding)
    logging.info('Removing borders using flood fill')
    binary_sauvola = flood_fill(binary_sauvola, (0, 0), 0)
    binary_sauvola = flood_fill(binary_sauvola, (0, 0), 1)
    
    # Save intermediate image after border removal
    if verbose:
        cv2.imwrite('verbose_after_border_removal.jpg', binary_sauvola)
        logging.info('Saved verbose image: verbose_after_border_removal.jpg')

    if noise_reduction:
        logging.info('Applying noise reduction')
        
        # Default noise reduction parameters
        default_params = {
            'hole_removal': 1.0,
            'opening_strength': 1.0,
            'closing_strength': 1.0,
            'median_strength': 1.0
        }
        
        # Merge with user parameters
        params = {**default_params, **noise_reduction}
        
        binary_cleaned = binary_sauvola.astype(bool)
        
        # Remove small holes (adjustable threshold)
        if params['hole_removal'] > 0:
            hole_threshold = int(holes_threshold * params['hole_removal'])
            binary_cleaned = remove_small_holes(binary_cleaned, area_threshold=hole_threshold)
            if params['hole_removal'] != 1.0:
                logging.info(f'Hole removal threshold: {hole_threshold}')
        
        # Remove small objects (noise specks) with adjustable kernel size
        if params['opening_strength'] > 0:
            kernel_size = max(1, int(2 * params['opening_strength']))
            kernel_small = np.ones((kernel_size, kernel_size), np.uint8)
            binary_cleaned = cv2.morphologyEx(binary_cleaned.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel_small)
            if params['opening_strength'] != 1.0:
                logging.info(f'Opening kernel size: {kernel_size}x{kernel_size}')
        else:
            binary_cleaned = binary_cleaned.astype(np.uint8) * 255
        
        # Close small gaps in text with adjustable kernel size
        if params['closing_strength'] > 0:
            kernel_size = max(1, int(3 * params['closing_strength']))
            kernel_close = np.ones((kernel_size, kernel_size), np.uint8)
            binary_cleaned = cv2.morphologyEx(binary_cleaned, cv2.MORPH_CLOSE, kernel_close)
            if params['closing_strength'] != 1.0:
                logging.info(f'Closing kernel size: {kernel_size}x{kernel_size}')
        
        # Remove isolated pixels with adjustable kernel size
        if params['median_strength'] > 0:
            kernel_size = max(3, int(3 * params['median_strength']))
            # Ensure kernel size is odd
            if kernel_size % 2 == 0:
                kernel_size += 1
            binary_cleaned = cv2.medianBlur(binary_cleaned, kernel_size)
            if params['median_strength'] != 1.0:
                logging.info(f'Median blur kernel size: {kernel_size}')
        
        binary_sauvola = binary_cleaned > 0
        logging.info('Noise reduction completed')

    logging.info('Binarization completed')
    return binary_sauvola.astype(bool)
