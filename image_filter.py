#!/usr/bin/env python3
"""
Image Filter Tool
Applies various filters to images with command line interface.
"""

import argparse
import sys
import time
import cv2
import numpy as np


def replace_black_filter(image):
    """Replace absolute black pixels (0) with most dominant color in surrounding area."""
    # Create mask for pixels that are exactly 0
    if len(image.shape) == 3:
        # Color image - check if all channels are 0
        black_mask = np.all(image == 0, axis=2)
    else:
        # Grayscale image
        black_mask = (image == 0)
    
    if not np.any(black_mask):
        return image  # No black pixels found
    
    result = image.copy()
    height, width = image.shape[:2]
    
    # Get coordinates of black pixels
    black_coords = np.where(black_mask)
    total_black_pixels = len(black_coords[0])
    
    print(f"Found {total_black_pixels} black pixels to process")
    
    # Progress tracking
    start_time = time.time()
    last_update_time = start_time
    
    # Process each black pixel
    for i in range(total_black_pixels):
        y, x = black_coords[0][i], black_coords[1][i]
        
        # Find replacement color using expanding radius (50px steps)
        replacement_color = find_dominant_color_around_pixel(image, x, y, height, width)
        
        # Replace the black pixel
        result[y, x] = replacement_color
        
        # Progress reporting every second
        current_time = time.time()
        if current_time - last_update_time >= 1.0:
            progress_percent = (i + 1) / total_black_pixels * 100
            elapsed_time = current_time - start_time
            estimated_total_time = elapsed_time / (i + 1) * total_black_pixels
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"Progress: {progress_percent:.1f}% ({i + 1}/{total_black_pixels}) - "
                  f"Elapsed: {elapsed_time:.1f}s, Estimated remaining: {remaining_time:.1f}s")
            last_update_time = current_time
    
    total_time = time.time() - start_time
    print(f"Completed processing {total_black_pixels} pixels in {total_time:.2f}s")
    
    return result


def find_dominant_color_around_pixel(image, x, y, height, width):
    """Find the most dominant non-black color around a pixel using expanding radius."""
    radius = 50  # Start with 50px radius
    
    while radius <= min(height, width):
        # Define square block bounds
        y_min = max(0, y - radius)
        y_max = min(height, y + radius + 1)
        x_min = max(0, x - radius)
        x_max = min(width, x + radius + 1)
        
        # Extract the region
        region = image[y_min:y_max, x_min:x_max]
        
        # Count non-black pixels
        if len(image.shape) == 3:
            non_black_mask = ~np.all(region == 0, axis=2)
        else:
            non_black_mask = region != 0
        
        total_pixels = region.shape[0] * region.shape[1]
        non_black_pixels = np.sum(non_black_mask)
        
        # Check if we have at least 50% non-black pixels
        if non_black_pixels >= total_pixels * 0.5:
            # Find most dominant color (excluding black)
            return get_dominant_color(region, non_black_mask)
        
        # Expand radius by 50px
        radius += 50
    
    # Fallback: return white if no suitable area found
    if len(image.shape) == 3:
        return [255, 255, 255]
    else:
        return 255


def get_dominant_color(region, non_black_mask):
    """Get the most dominant color from a region, excluding black pixels."""
    if len(region.shape) == 3:
        # Color image
        non_black_pixels = region[non_black_mask]
        if len(non_black_pixels) == 0:
            return [255, 255, 255]  # Fallback to white
        
        # Use average color as approximation of dominant color
        # For true dominant color, we'd need histogram analysis which is more complex
        dominant_color = np.mean(non_black_pixels, axis=0).astype(np.uint8)
        return dominant_color
    else:
        # Grayscale image
        non_black_pixels = region[non_black_mask]
        if len(non_black_pixels) == 0:
            return 255  # Fallback to white
        
        # Use average value
        dominant_color = np.mean(non_black_pixels).astype(np.uint8)
        return dominant_color


def enhance_local_contrast_filter(image, radius):
    """Enhance local contrast using median blur subtraction with mask preservation."""
    start_time = time.time()
    total_steps = 6
    current_step = 0
    
    # Step 1: Create mask for pixels that are exactly 0 (to be preserved)
    current_step += 1
    print(f"Step {current_step}/{total_steps}: Creating mask for preserved pixels...")
    step_start = time.time()
    
    if len(image.shape) == 3:
        # Color image - check if all channels are 0
        mask = np.all(image == 0, axis=2)
    else:
        # Grayscale image
        mask = (image == 0)
    
    print(f"Found {np.sum(mask)} masked pixels (value 0) to preserve")
    print(f"Using median blur radius: {radius}")
    
    # Convert radius to kernel size (must be odd)
    kernel_size = 2 * radius + 1
    print(f"Median blur kernel size: {kernel_size}")
    print(f"Step {current_step} completed in {time.time() - step_start:.2f}s")
    
    # Step 2: Create working copy and fill masked pixels
    current_step += 1
    print(f"Step {current_step}/{total_steps}: Preparing image data...")
    step_start = time.time()
    
    result = image.copy().astype(np.float32)
    print(f"Step {current_step} completed in {time.time() - step_start:.2f}s")
    
    # Step 3: Apply median blur while preserving masked areas
    current_step += 1
    print(f"Step {current_step}/{total_steps}: Applying median blur...")
    step_start = time.time()
    
    if len(image.shape) == 3:
        # Color image
        blurred = np.zeros_like(result)
        total_channels = image.shape[2]
        for channel in range(total_channels):
            print(f"  Processing channel {channel + 1}/{total_channels}...")
            # Create temporary image with masked pixels filled with surrounding values
            temp_channel = image[:, :, channel].copy().astype(np.float32)
            temp_channel = fill_masked_pixels(temp_channel, mask)
            
            # Apply median blur
            blurred_channel = cv2.medianBlur(temp_channel.astype(np.uint8), kernel_size)
            blurred[:, :, channel] = blurred_channel.astype(np.float32)
    else:
        # Grayscale image
        temp_image = image.copy().astype(np.float32)
        temp_image = fill_masked_pixels(temp_image, mask)
        blurred = cv2.medianBlur(temp_image.astype(np.uint8), kernel_size).astype(np.float32)
    
    print(f"Step {current_step} completed in {time.time() - step_start:.2f}s")
    
    # Step 4: Subtract blurred from original (high-pass filter)
    current_step += 1
    print(f"Step {current_step}/{total_steps}: Computing high-pass filter (original - blurred)...")
    step_start = time.time()
    
    # Debug: Show original and blurred value ranges
    if len(image.shape) == 3:
        orig_min, orig_max = np.min(result[~mask]), np.max(result[~mask])
        blur_min, blur_max = np.min(blurred[~mask]), np.max(blurred[~mask])
    else:
        orig_min, orig_max = np.min(result[~mask]), np.max(result[~mask])
        blur_min, blur_max = np.min(blurred[~mask]), np.max(blurred[~mask])
    
    print(f"Original image value range: {orig_min:.1f} to {orig_max:.1f}")
    print(f"Blurred image value range: {blur_min:.1f} to {blur_max:.1f}")
    
    contrast_enhanced = result - blurred
    
    # Normalize high-pass result to 1..254 (keep mask 0)
    non_masked_pixels = contrast_enhanced[~mask]
    if len(non_masked_pixels) > 0:
        min_val = np.min(non_masked_pixels)
        max_val = np.max(non_masked_pixels)
        if max_val > min_val:
            contrast_enhanced[~mask] = ((contrast_enhanced[~mask] - min_val) /
                                        (max_val - min_val)) * 253 + 1
        else:
            contrast_enhanced[~mask] = 128
    else:
        contrast_enhanced[~mask] = 128
    
    # Debug: Show high-pass filter result range
    if len(image.shape) == 3:
        hp_min, hp_max = np.min(contrast_enhanced[~mask]), np.max(contrast_enhanced[~mask])
    else:
        hp_min, hp_max = np.min(contrast_enhanced[~mask]), np.max(contrast_enhanced[~mask])
    
    print(f"High-pass filter result range: {hp_min:.1f} to {hp_max:.1f}")
    print(f"Step {current_step} completed in {time.time() - step_start:.2f}s")
    
    # Step 5: Global contrast stretch based on central circle statistics
    current_step += 1
    print(f"Step {current_step}/{total_steps}: Global contrast stretch based on central circle...")
    step_start = time.time()

    h, w = image.shape[:2]
    radius = min(h, w) // 8
    center_y, center_x = h // 2, w // 2

    # Build circular mask for central region
    y, x = np.ogrid[:h, :w]
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    circle_mask = dist <= radius

    # Collect min/max inside circle (excluding mask)
    valid = circle_mask & ~mask
    if np.any(valid):
        min_val = np.min(contrast_enhanced[valid])
        max_val = np.max(contrast_enhanced[valid])
    else:
        min_val, max_val = 0, 255

    # Stretch contrast: map min_val→1, max_val→254
    if max_val > min_val:
        scale = 253 / (max_val - min_val)
        offset = 1 - scale * min_val
        stretched = contrast_enhanced * scale + offset
    else:
        stretched = np.full_like(contrast_enhanced, 128)

    # Clip to 1..254 for all non-masked pixels
    stretched = np.clip(stretched, 1, 254)

    # Restore masked pixels (keep 0)
    stretched[mask] = 0

    # Convert to uint8
    result = np.clip(stretched, 0, 255).astype(np.uint8)

    print(f"Step {current_step} completed in {time.time() - step_start:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Local contrast enhancement completed in {total_time:.2f}s")
    return result


def fill_masked_pixels(image, mask):
    """Fill masked pixels with average of surrounding non-masked pixels."""
    result = image.copy()
    
    # Get coordinates of masked pixels
    masked_coords = np.where(mask)
    
    if len(masked_coords[0]) == 0:
        return result
    
    height, width = image.shape
    
    # For each masked pixel, find replacement value
    for i in range(len(masked_coords[0])):
        y, x = masked_coords[0][i], masked_coords[1][i]
        
        # Look in expanding neighborhoods for non-masked pixels
        for radius in range(1, min(height, width) // 2):
            y_min = max(0, y - radius)
            y_max = min(height, y + radius + 1)
            x_min = max(0, x - radius)
            x_max = min(width, x + radius + 1)
            
            # Extract neighborhood
            neighborhood = image[y_min:y_max, x_min:x_max]
            neighborhood_mask = mask[y_min:y_max, x_min:x_max]
            
            # Get non-masked pixels in neighborhood
            non_masked_pixels = neighborhood[~neighborhood_mask]
            
            if len(non_masked_pixels) > 0:
                # Use average of non-masked pixels
                result[y, x] = np.mean(non_masked_pixels)
                break
        else:
            # Fallback: use overall image average (excluding masked pixels)
            non_masked_global = image[~mask]
            if len(non_masked_global) > 0:
                result[y, x] = np.mean(non_masked_global)
            else:
                result[y, x] = 128  # Neutral gray
    
    return result


def bright_threshold_filter(image, kernel_divisor, percentile):
    """Apply bright threshold filter using median blur and percentile-based thresholding."""
    height, width = image.shape[:2]
    
    # Calculate kernel size as 1/nth of width
    kernel_size = max(1, width // kernel_divisor)
    # Ensure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    print(f"Image dimensions: {width}x{height}")
    print(f"Kernel size: {kernel_size} (width/{kernel_divisor})")
    print(f"Using {percentile}% percentile threshold")
    
    # Apply median blur
    print("Applying median blur...")
    if len(image.shape) == 3:
        blurred = cv2.medianBlur(image, kernel_size)
    else:
        blurred = cv2.medianBlur(image, kernel_size)
    
    result = image.copy()
    
    # Process in 50px blocks
    block_size = 50
    total_blocks = ((height + block_size - 1) // block_size) * ((width + block_size - 1) // block_size)
    processed_blocks = 0
    
    print(f"Processing in {block_size}x{block_size} blocks...")
    start_time = time.time()
    last_update_time = start_time
    
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            # Define block bounds
            y_end = min(y + block_size, height)
            x_end = min(x + block_size, width)
            
            # Extract blocks from both original and blurred images
            original_block = image[y:y_end, x:x_end]
            blurred_block = blurred[y:y_end, x:x_end]
            
            # Process block for bright threshold
            result[y:y_end, x:x_end] = process_block_bright_threshold(
                original_block, blurred_block, percentile
            )
            
            processed_blocks += 1
            
            # Progress reporting every second
            current_time = time.time()
            if current_time - last_update_time >= 1.0:
                progress_percent = processed_blocks / total_blocks * 100
                elapsed_time = current_time - start_time
                estimated_total_time = elapsed_time / processed_blocks * total_blocks
                remaining_time = estimated_total_time - elapsed_time
                
                print(f"Progress: {progress_percent:.1f}% ({processed_blocks}/{total_blocks}) - "
                      f"Elapsed: {elapsed_time:.1f}s, Estimated remaining: {remaining_time:.1f}s")
                last_update_time = current_time
    
    total_time = time.time() - start_time
    print(f"Completed processing {total_blocks} blocks in {total_time:.2f}s")
    
    return result


def process_block_bright_threshold(original_block, blurred_block, percentile):
    """Process a single block for bright threshold filter."""
    if len(blurred_block.shape) == 3:
        # Color image - convert to grayscale for intensity calculation
        blurred_gray = cv2.cvtColor(blurred_block, cv2.COLOR_BGR2GRAY)
    else:
        blurred_gray = blurred_block
    
    # Get all pixel intensities in the blurred block
    pixel_intensities = blurred_gray.flatten()
    
    # Calculate the percentile threshold
    threshold_value = np.percentile(pixel_intensities, percentile)
    
    # Find pixels above the percentile threshold
    bright_mask = blurred_gray >= threshold_value
    
    if not np.any(bright_mask):
        # No bright pixels found, return original block
        return original_block
    
    # Get the darkest color among the bright pixels
    if len(blurred_block.shape) == 3:
        # Color image
        bright_pixels = blurred_gray[bright_mask]
        darkest_bright_value = np.min(bright_pixels)
        
        # Create mask for pixels brighter than the darkest bright value in original block
        if len(original_block.shape) == 3:
            original_gray = cv2.cvtColor(original_block, cv2.COLOR_BGR2GRAY)
        else:
            original_gray = original_block
        
        mask_to_whiten = original_gray >= darkest_bright_value
    else:
        # Grayscale image
        bright_pixels = blurred_gray[bright_mask]
        darkest_bright_value = np.min(bright_pixels)
        
        # Create mask for pixels brighter than the darkest bright value in original block
        mask_to_whiten = original_block >= darkest_bright_value
    
    # Apply whitening to the result
    result_block = original_block.copy()
    if len(result_block.shape) == 3:
        # Color image - set all channels to 254
        result_block[mask_to_whiten] = [254, 254, 254]
    else:
        # Grayscale image
        result_block[mask_to_whiten] = 254
    
    return result_block


def apply_filter(image, filter_name, **kwargs):
    """Apply the specified filter to the image."""
    if filter_name == 'replace_black':
        return replace_black_filter(image)
    elif filter_name == 'bright_threshold':
        return bright_threshold_filter(image, kwargs['kernel_divisor'], kwargs['percentile'])
    elif filter_name == 'enhance_local_contrast':
        return enhance_local_contrast_filter(image, kwargs['radius'])
    else:
        raise ValueError(f"Unknown filter: {filter_name}")


def main():
    parser = argparse.ArgumentParser(
        description='Image Filter Tool - Apply various filters to images',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('-i', '--input', required=True,
                        help='Input image file path')
    parser.add_argument('-o', '--output', required=True,
                        help='Output image file path')
    
    # Filter options (mutually exclusive)
    filter_group = parser.add_mutually_exclusive_group(required=True)
    filter_group.add_argument('--replace-black', action='store_true',
                              help='Replace absolute black pixels (0) with most dominant color in surrounding area')
    filter_group.add_argument('--bright-threshold', action='store_true',
                              help='Apply bright threshold filter using median blur and percentile-based thresholding')
    filter_group.add_argument('--enhance-local-contrast', action='store_true',
                              help='Enhance local contrast using median blur subtraction with mask preservation')
    
    # Parameters for bright-threshold filter
    parser.add_argument('--kernel-divisor', type=int, default=10,
                        help='Kernel size divisor (kernel = width/divisor) for median blur (default: 10)')
    parser.add_argument('--percentile', type=float, default=80.0,
                        help='Percentile threshold for bright pixels (default: 80.0)')
    
    # Parameters for enhance-local-contrast filter
    parser.add_argument('--radius', type=int, default=5,
                        help='Radius for median blur in enhance-local-contrast filter (default: 5)')
    
    # Add help for filters
    parser.epilog = """
Available Filters:
  --replace-black           Replace absolute black pixels (0) with most dominant color
                            in surrounding area. Uses expanding radius (50px steps) until
                            at least 50% non-black pixels are found.
  
  --bright-threshold        Apply bright threshold filter using median blur and percentile-based
                            thresholding. Parameters: --kernel-divisor, --percentile
  
  --enhance-local-contrast  Enhance local contrast using median blur subtraction with mask
                            preservation. Uses color 0 as mask (keeps unmodified).
                            Parameters: --radius

Examples:
  python image_filter.py -i input.jpg -o output.jpg --replace-black
  python image_filter.py -i input.jpg -o output.jpg --bright-threshold --kernel-divisor 8 --percentile 85
  python image_filter.py -i input.jpg -o output.jpg --enhance-local-contrast --radius 10
"""
    
    args = parser.parse_args()
    
    # Determine which filter to apply
    filter_name = None
    filter_kwargs = {}
    
    if args.replace_black:
        filter_name = 'replace_black'
    elif args.bright_threshold:
        filter_name = 'bright_threshold'
        filter_kwargs = {
            'kernel_divisor': args.kernel_divisor,
            'percentile': args.percentile
        }
    elif args.enhance_local_contrast:
        filter_name = 'enhance_local_contrast'
        filter_kwargs = {
            'radius': args.radius
        }
    
    if filter_name is None:
        parser.error('No filter specified. Use --help to see available filters.')
    
    try:
        # Print input/output file names
        print(f"Input file: {args.input}")
        print(f"Output file: {args.output}")
        
        # Load image
        image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
        if image is None:
            print(f"Error: Could not load image from {args.input}")
            sys.exit(1)
        
        print(f"Applying filter: {filter_name}")
        
        # Apply the filter
        filtered_image = apply_filter(image, filter_name, **filter_kwargs)
        
        # Save the result
        success = cv2.imwrite(args.output, filtered_image)
        if not success:
            print(f"Error: Could not save image to {args.output}")
            sys.exit(1)
        
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
