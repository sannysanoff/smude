__author__ = "Simon Waloschek"

import logging
import os
import argparse

import cv2 as cv
import numpy as np
import requests
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from skimage.color import gray2rgb
from skimage.io import imread, imsave
from tqdm import tqdm

from .binarize import binarize
from .model import load_model
from .mrcdi import mrcdi
from .roi import extract_roi_mask, get_border
from .utils import get_logger

# --- Local-contrast enhancement helper ---------------------------------
def enhance_local_contrast_filter(image, radius, *, verbose=False, threshold: int = 128):
    """Enhance local contrast using median-blur subtraction with mask preservation."""
    import cv2
    import numpy as np
    import time
    import os

    start_time = time.time()
    total_steps = 6
    current_step = 0

    # helper to save any ndarray
    def _save_step(arr, name):
        if not verbose:
            return
        out_name = f'verbose_{name}.jpg'
        # ensure uint8 BGR for cv2.imwrite
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        if len(arr.shape) == 2:           # grayscale
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(out_name, arr)
        logging.info(f'Saved verbose image: {out_name}')

    # Step 1: mask for pixels == 0
    current_step += 1
    mask = (image == 0) if image.ndim == 2 else np.any(image == 0, axis=2)
    _save_step(mask.astype(np.uint8) * 255, f'01_mask')

    kernel_size = 2 * radius + 1

    # Step 2: grayscale + normalize non-masked to 1..255
    current_step += 1
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.astype(np.uint8)
    gray = gray.astype(np.float32)
    gray[mask] = 0.0
    non_masked = gray[~mask]
    if len(non_masked):
        g_min, g_max = non_masked.min(), non_masked.max()
        gray[~mask] = 1.0 + 254.0 * (gray[~mask] - g_min) / (g_max - g_min) if g_max > g_min else 128.0
    else:
        gray[~mask] = 128.0
    _save_step(gray, f'02_gray_normalized')

    # Step 3: median blur
    current_step += 1
    temp = gray.copy()
    # Use image-width based kernel size but cap at 99 and min 31
    kernel_size = min(99, max(31, int(image.shape[1] / 10)))
    if kernel_size % 2 == 0:          # ensure odd
        kernel_size += 1
    kernel_size = min(99, kernel_size)    # final safeguard
    logging.info(f'Median blur kernel size: {kernel_size}')
    # Fill masked pixels with the median of the non-masked ones
    median_val = np.median(gray[~mask]) if np.any(~mask) else 128.0
    temp[mask] = median_val
    blurred = cv2.medianBlur(temp.astype(np.uint8), kernel_size).astype(np.float32)
    blurred[mask] = 0.0
    _save_step(blurred, f'03_blurred')

    # Step 4: high-pass filter & normalize 1..254
    current_step += 1
    contrast_enhanced = gray - blurred
    non_masked_pixels = contrast_enhanced[~mask]
    if len(non_masked_pixels):
        min_val, max_val = non_masked_pixels.min(), non_masked_pixels.max()
        if max_val > min_val:
            contrast_enhanced[~mask] = ((contrast_enhanced[~mask] - min_val) / (max_val - min_val)) * 253 + 1
        else:
            contrast_enhanced[~mask] = 128
    else:
        contrast_enhanced[~mask] = 128
    _save_step(contrast_enhanced, f'04_contrast_enhanced')

    # Step 5: threshold
    current_step += 1
    # use the threshold parameter provided to the function
    binary = np.where(contrast_enhanced < threshold, 1, 254)
    binary[mask] = 0
    _save_step(binary, f'05_binary')

    # Convert back to 3-channel RGB so downstream code expects RGB
    binary = binary.astype(np.uint8)
    binary = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    _save_step(binary, f'06_final_rgb')
    return binary
# -----------------------------------------------------------------------

# Initialize logging
logger = get_logger()


class Smude():
    def __init__(self, use_gpu: bool = False, binarize_output: bool = True, verbose: bool = False, noise_reduction: dict = None, max_dist: float = 40.0, threshold: int = 128, sauvola_k: float = 0.25, skip_border_removal: bool = False, grow: int = 0, spline_threshold: int = 80, pad: int = 0):
        """
        Instantiate new Smude object for sheet music dewarping.

        Parameters
        ----------
        use_gpu : bool, optional
            Flag if GPU should be used, by default False.
        binarize_output : bool, optional
            Flag whether the output should be binarized, by default True.
        verbose : bool, optional
            Flag whether to enable verbose output with intermediate images, by default False.
        noise_reduction : dict, optional
            Dictionary controlling noise reduction aggressiveness during binarization.
            Keys: 'hole_removal', 'opening_strength', 'closing_strength', 'median_strength'
            Values: float multipliers (1.0 = default strength, 0 = disabled), by default None.
        threshold : int, optional
            Threshold value for binarization (default: 128).
        max_dist : float, optional
            Maximum allowed distance between staff lines for detection (default: 40.0).
        pad : int, optional
            Amount of padding (in pixels) to add to input image before processing (default: 0).
        """

        super().__init__()
        self.use_gpu = use_gpu
        self.binarize_output = binarize_output
        self.verbose = verbose
        self.noise_reduction = noise_reduction
        self.threshold = threshold
        self.max_dist = max_dist
        self.sauvola_k = sauvola_k
        self.skip_border_removal = skip_border_removal
        self.grow = grow
        self.spline_threshold = spline_threshold
        self.pad = pad

        # Load Deep Learning model
        dirname = os.path.dirname(__file__)
        checkpoint_path = os.path.join(dirname, 'model.ckpt')
        if not os.path.exists(checkpoint_path):
            print('First run. Downloading model...')
            url = 'https://github.com/sonovice/smude/releases/download/v0.1.0/model.ckpt'
            response = requests.get(url, stream=True, allow_redirects=True)
            total_size_in_bytes= int(response.headers.get('content-length', 0))
            block_size = 1024 #1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(checkpoint_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("Error: Model could not be downloaded.")
                exit(1)

        self.model = load_model(checkpoint_path)
        if self.use_gpu:
            self.model = self.model.cuda()
        self.model.freeze()

        # Define transformations on input image
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(900),
            transforms.Grayscale(3),
            transforms.ToTensor()
        ])


    def process(self, image: np.ndarray, optimize_f: bool = False) -> np.ndarray:
        """
        Extract region of interest from sheet music image and dewarp it.

        Parameters
        ----------
        image : np.ndarray
            Input sheet music image.
        use_gpu : bool
            Flag whether to use GPU/CUDA to speed up the process.

        Returns
        -------
        np.ndarray
            Dewarped sheet music image.
        """

        # Add padding if requested
        if self.pad > 0:
            logging.info(f'Adding {self.pad} pixels of padding to input image')
            pad_width = self.pad
            if len(image.shape) == 3:
                padding = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))
            else:
                padding = ((pad_width, pad_width), (pad_width, pad_width))
            image = np.pad(image, padding, mode='constant', constant_values=255)
            
        original_shape = image.shape[:2]

        if len(image.shape) < 3:
            image = gray2rgb(image)

        if self.verbose:
            self._save_verbose_image(image, 'input')

        logging.info('Extracting ROI...')
        roi_mask, mask_ratio = extract_roi_mask(image)

        if self.verbose:
            self._save_verbose_image(roi_mask * 255, 'roi_mask')

        # Repeat mask for each RGB channel
        mask_3c = np.broadcast_to(roi_mask[..., None], roi_mask.shape + (3,))
        # Shift non-masked pixels to 2..255 so 0 remains reserved for the mask
        result = image.astype(np.float32)
        result = (result / 255.0) * 253 + 2          # scale 0..255 → 2..255
        result = result * mask_3c                    # masked pixels stay 0
        result = result.astype(np.uint8)

        if self.verbose:
            self._save_verbose_image(result, 'masked_roi')

        if self.verbose:
            self._save_verbose_image(result, 'before_enhance')

        logging.info('Enhancing local contrast (step 10)...')
        enhanced = enhance_local_contrast_filter(
            result, radius=5, threshold=self.threshold, verbose=self.verbose
        )
        if self.verbose:
            self._save_verbose_image(enhanced, 'enhanced_local_contrast')

        logging.info('Flood filling edge black pixels...')
        # Create mask for flood fill (2 pixels larger than image)
        mask = np.zeros((enhanced.shape[0] + 2, enhanced.shape[1] + 2), np.uint8)
        
        # Fill border of mask with 255 to allow flood fill from edges
        mask[0, :] = 255
        mask[-1, :] = 255
        mask[:, 0] = 255
        mask[:, -1] = 255
        
        # Flood fill all black pixels connected to edges
        for y in range(enhanced.shape[0]):
            for x in [0, enhanced.shape[1]-1]:
                if enhanced[y, x].sum() == 0:  # If pixel is black
                    cv.floodFill(enhanced, mask, (x, y), (255, 255, 255), 
                                flags=cv.FLOODFILL_FIXED_RANGE)
        
        for x in range(enhanced.shape[1]):
            for y in [0, enhanced.shape[0]-1]:
                if enhanced[y, x].sum() == 0:  # If pixel is black
                    cv.floodFill(enhanced, mask, (x, y), (255, 255, 255), 
                                flags=cv.FLOODFILL_FIXED_RANGE)
        
        if self.verbose:
            self._save_verbose_image(enhanced, 'after_flood_fill')

        logging.info('Binarizing...')
        # Binarize ROI
        binarized = binarize(enhanced, noise_reduction=self.noise_reduction, threshold=self.threshold, verbose=self.verbose, sauvola_k=self.sauvola_k, skip_border_removal=self.skip_border_removal)

        if self.verbose:
            self._save_verbose_image(binarized * 255, 'binarized')

        # Remove borders
        x_start, x_end, y_start, y_end = get_border(binarized)
        binarized = binarized[x_start:x_end, y_start:y_end]

        # Add 5% width border
        pad_width = int(binarized.shape[0] * 0.05)
        binarized = np.pad(binarized, pad_width=pad_width, mode='constant', constant_values=1)

        if self.verbose:
            self._save_verbose_image(binarized * 255, 'padded')

        binarized_torch = torch.from_numpy(binarized).float()

        # Resize and convert binary image to grayscale torch tensor
        grayscale = self.transforms(binarized_torch).float()

        if self.verbose:
            grayscale_np = (grayscale.numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
            self._save_verbose_image(grayscale_np, 'preprocessed_for_unet')

        if self.grow > 0:
            logging.info(f'Growing black pixels by {self.grow} pixels...')
            # Create a copy to work with
            grown = grayscale_np.copy()
            # Find all non-white pixels
            black_pixels = np.where(grayscale_np < 255)
            # For each black pixel, grow in manhattan distance
            for y, x in zip(black_pixels[0], black_pixels[1]):
                # Create a diamond-shaped kernel for manhattan distance
                for dy in range(-self.grow, self.grow + 1):
                    for dx in range(-self.grow, self.grow + 1):
                        if abs(dx) + abs(dy) <= self.grow:  # Manhattan distance
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < grayscale_np.shape[0] and 0 <= nx < grayscale_np.shape[1]:
                                grown[ny, nx] = 0
            grayscale_np = grown
            if self.verbose:
                self._save_verbose_image(grayscale_np, 'grown_black_pixels')

        logging.info('Extracting features...')

        # Move to GPU
        if self.use_gpu:
            grayscale = grayscale.cuda()

        # Run inference
        output = self.model(grayscale.unsqueeze(0)).cpu()
        classes = torch.argmax(F.softmax(output[0], dim=0), dim=0)

        # Convert images to correct data types
        grayscale = (grayscale.cpu().numpy().transpose([1, 2, 0]) * 255).astype(np.uint8)
        background = (1.0 * (classes == 0).numpy() * 255).astype(np.uint8)
        upper = (1.0 * (classes == 1).numpy() * 255).astype(np.uint8)
        lower = (1.0 * (classes == 2).numpy() * 255).astype(np.uint8)
        barlines = (1.0 * (classes == 3).numpy() * 255).astype(np.uint8)
        binarized = (binarized * 255).astype(np.uint8)

        if self.verbose:
            self._save_verbose_image(background, 'unet_background')
            self._save_verbose_image(upper, 'unet_upper')
            self._save_verbose_image(lower, 'unet_lower')
            self._save_verbose_image(barlines, 'unet_barlines')

        logging.info('Dewarping...')

        # Dewarp output
        cols, rows = mrcdi(
            input_img = grayscale,
            barlines_img = barlines,
            upper_img = upper,
            lower_img = lower,
            background_img = background,
            original_img = binarized,
            optimize_f = optimize_f,
            verbose = self.verbose,
            max_dist = self.max_dist,
            spline_threshold = self.spline_threshold
        )
        
        if self.binarize_output:
            dewarped = cv.remap(binarized, cols, rows, cv.INTER_CUBIC, None, cv.BORDER_CONSTANT, 255)
            # Remove border
            x_start, x_end, y_start, y_end = get_border(dewarped)
            dewarped = dewarped[x_start:x_end, y_start:y_end]

            # Add 5% min(width, height) border
            smaller = min(*dewarped.shape)
            dewarped = np.pad(dewarped, pad_width=int(smaller * 0.05), mode='constant', constant_values=255)
        else:
            # TODO rework the image manipulation part here
            # Remove borders
            image = image[x_start:x_end, y_start:y_end]
            dewarped = []
            # Do stuff for each channel individually
            for c in range(image.shape[2]):
                # Add border
                channel = np.pad(image[:, :, c], pad_width=pad_width, mode='constant', constant_values=255)
                # Dewarp
                channel = cv.remap(channel, cols, rows, cv.INTER_CUBIC, None, cv.BORDER_CONSTANT, 255)
                # Remove border again
                channel = channel[pad_width:-pad_width, pad_width:-pad_width]
                
                border_cols, border_rows = np.where(channel < 255)
                x_start = np.min(border_cols)
                x_end = np.max(border_cols) + 1
                y_start = np.min(border_rows)
                y_end = np.max(border_rows) + 1
                                
                channel = channel[x_start:x_end, y_start:y_end]
                
                dewarped.append(channel)
            dewarped = np.stack(dewarped, axis=2)
        
        # Remove padding if it was added
        if self.pad > 0:
            logging.info(f'Removing {self.pad} pixels of padding from final image')
            if self.binarize_output:
                dewarped = dewarped[self.pad:-self.pad, self.pad:-self.pad]
            else:
                if len(dewarped.shape) == 3:
                    dewarped = dewarped[self.pad:-self.pad, self.pad:-self.pad, :]
                else:
                    dewarped = dewarped[self.pad:-self.pad, self.pad:-self.pad]
        
        if self.verbose:
            self._save_verbose_image(dewarped, 'final_result')
            
        return dewarped

    def _save_verbose_image(self, image: np.ndarray, step_name: str):
        """Save intermediate image with descriptive name only."""
        filename = f'verbose_{step_name}.jpg'
        
        try:
            # Ensure image is in proper format for saving
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            
            # Convert to numpy array if needed
            image = np.asarray(image)
            
            # Handle different image formats and ensure proper data types
            if len(image.shape) == 3 and image.shape[2] == 3:
                # RGB image - ensure uint8
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = np.clip(image, 0, 255).astype(np.uint8)
                cv.imwrite(filename, cv.cvtColor(image, cv.COLOR_RGB2BGR))
            elif len(image.shape) == 3 and image.shape[2] == 1:
                # Single channel with extra dimension - squeeze and save as grayscale
                image = image.squeeze()
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = np.clip(image, 0, 255).astype(np.uint8)
                cv.imwrite(filename, image)
            elif len(image.shape) == 2:
                # Grayscale image
                if image.dtype != np.uint8:
                    if image.max() <= 1.0:
                        image = (image * 255).astype(np.uint8)
                    else:
                        image = np.clip(image, 0, 255).astype(np.uint8)
                cv.imwrite(filename, image)
            else:
                # Handle unexpected shapes - flatten to 2D if possible
                if image.size > 0:
                    if len(image.shape) > 2:
                        # Try to convert multi-dimensional to 2D
                        image = np.squeeze(image)
                        if len(image.shape) > 2:
                            # Take first channel if still multi-dimensional
                            image = image[:, :, 0] if image.shape[2] > 0 else image.flatten().reshape(1, -1)
                    
                    if image.dtype != np.uint8:
                        if image.max() <= 1.0:
                            image = (image * 255).astype(np.uint8)
                        else:
                            image = np.clip(image, 0, 255).astype(np.uint8)
                    cv.imwrite(filename, image)
                else:
                    logging.warning(f'Cannot save empty image: {filename}')
                    return
            
            logging.info(f'Saved verbose image: {filename}')
            
        except Exception as e:
            logging.error(f'Failed to save verbose image {filename}: {str(e)}')
            logging.error(f'Image shape: {image.shape}, dtype: {image.dtype}')

def main():
    parser = argparse.ArgumentParser(description='Dewarp and binarize sheet music images.')
    parser.add_argument('infile', help='Specify the input image file path')
    parser.add_argument('-o', '--outfile', help='Specify the output image file path (default: result.png)', default='result.png')
    parser.add_argument('--no-binarization', help='Deactivate binarization of output (default: enabled)', action='store_false')
    parser.add_argument('--use-gpu', help='Use GPU acceleration for neural network inference (default: disabled)', action='store_true')
    parser.add_argument('--verbose', help='Enable verbose logging with intermediate image outputs (default: disabled)', action='store_true')
    parser.add_argument('--noise-reduction', 
                       help='Noise reduction settings as comma-separated key=value pairs. Available parameters: hole_removal: Hole removal threshold multiplier (range: 0.0-5.0, default: 1.0), opening_strength: Opening operation kernel size multiplier (range: 0.0-5.0, default: 1.0), closing_strength: Closing operation kernel size multiplier (range: 0.0-5.0, default: 1.0), median_strength: Median filter kernel size multiplier (range: 0.0-5.0, default: 1.0). Example: hole_removal=1.5,opening_strength=2.0,median_strength=0.8. Set to 0 to disable specific operations', 
                       default=None)
    parser.add_argument('--max-dist', type=float, default=40.0, help='Maximum allowed distance between staff lines for detection (default: 40.0)')
    parser.add_argument('--threshold', type=int, default=128, help='Threshold value for binarization (default: 128)')
    parser.add_argument('--sauvola-k', type=float, default=0.25, help='Sauvola algorithm k parameter for niBlackThreshold (default: 0.25)')
    parser.add_argument('--skip-border-removal', help='Skip border removal using flood fill', action='store_true')
    parser.add_argument('--grow', type=int, default=0, help='Grow black pixels by n pixels in manhattan distance to remove tiny white dots (default: 0)')
    parser.add_argument('--spline-threshold', type=int, default=80, help='Keep most smooth splines (1-99%%, default: 80)')
    parser.add_argument('--pad', type=int, default=0, help='Pad input image with n pixels of white space (default: 0)')

    args = parser.parse_args()

    # Parse noise reduction parameters
    noise_reduction = None
    if args.noise_reduction:
        noise_reduction = {}
        try:
            for param in args.noise_reduction.split(','):
                key, value = param.strip().split('=')
                noise_reduction[key.strip()] = float(value.strip())
        except ValueError:
            print("Error: Invalid noise reduction format. Use key=value pairs separated by commas.")
            exit(1)

    smude = Smude(use_gpu=args.use_gpu, binarize_output=args.no_binarization, verbose=args.verbose, noise_reduction=noise_reduction, max_dist=args.max_dist, threshold=args.threshold, sauvola_k=args.sauvola_k, skip_border_removal=args.skip_border_removal, grow=args.grow, spline_threshold=args.spline_threshold, pad=args.pad)

    image = imread(args.infile)
    result = smude.process(image)
    imsave(args.outfile, result)
