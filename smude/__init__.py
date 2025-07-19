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
def enhance_local_contrast_filter(image, radius, **kwargs):
    """Enhance local contrast using median-blur subtraction with mask preservation."""
    import cv2
    import numpy as np
    import time

    start_time = time.time()
    total_steps = 6
    current_step = 0

    # Step 1: mask for pixels == 0
    current_step += 1
    mask = (image == 0) if image.ndim == 2 else np.any(image == 0, axis=2)
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

    # Step 3: median blur
    current_step += 1
    temp = gray.copy()
    temp[mask] = np.mean(gray[~mask]) if np.any(~mask) else 128.0
    blurred = cv2.medianBlur(temp.astype(np.uint8), kernel_size).astype(np.float32)
    blurred[mask] = 0.0

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

    # Step 5: threshold
    current_step += 1
    threshold = kwargs.get('threshold', 128)
    binary = np.where(contrast_enhanced < threshold, 1, 254)
    binary[mask] = 0
    return binary.astype(np.uint8)
# -----------------------------------------------------------------------

# Initialize logging
logger = get_logger()


class Smude():
    def __init__(self, use_gpu: bool = False, binarize_output: bool = True, verbose: bool = False, noise_reduction: dict = None):
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
        checkpoint_path : str, optional
            Path to a trained U-Net model, by default the included 'model.ckpt'.
        """

        super().__init__()
        self.use_gpu = use_gpu
        self.binarize_output = binarize_output
        self.verbose = verbose
        self.noise_reduction = noise_reduction
        self.step_counter = 9          # was 0

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
        # Obtain masked result image
        result = image * mask_3c

        if self.verbose:
            self._save_verbose_image(result, 'masked_roi')

        logging.info('Enhancing local contrast (step 10)...')
        enhanced = enhance_local_contrast_filter(
            result, radius=5, threshold=128
        )
        if self.verbose:
            self._save_verbose_image(enhanced, 'enhanced_local_contrast')

        logging.info('Binarizing...')
        # Binarize ROI
        binarized = binarize(enhanced, noise_reduction=self.noise_reduction)

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
            verbose = self.verbose
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
                       help='''Noise reduction settings as comma-separated key=value pairs. 
                            Available parameters:
                            - hole_removal: Hole removal threshold multiplier (range: 0.0-5.0, default: 1.0)
                            - opening_strength: Opening operation kernel size multiplier (range: 0.0-5.0, default: 1.0) 
                            - closing_strength: Closing operation kernel size multiplier (range: 0.0-5.0, default: 1.0)
                            - median_strength: Median filter kernel size multiplier (range: 0.0-5.0, default: 1.0)
                            Example: hole_removal=1.5,opening_strength=2.0,median_strength=0.8
                            Set to 0 to disable specific operations''', 
                       default=None)
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

    smude = Smude(use_gpu=args.use_gpu, binarize_output=args.no_binarization, verbose=args.verbose, noise_reduction=noise_reduction)

    image = imread(args.infile)
    result = smude.process(image)
    imsave(args.outfile, result)
