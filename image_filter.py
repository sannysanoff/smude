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


def apply_filter(image, filter_name, **kwargs):
    """Apply the specified filter to the image."""
    if filter_name == 'replace_black':
        return replace_black_filter(image)
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
    
    # Add help for filters
    parser.epilog = """
Available Filters:
  --replace-black    Replace absolute black pixels (0) with most dominant color
                     in surrounding area. Uses expanding radius (50px steps) until
                     at least 50% non-black pixels are found.

Examples:
  python image_filter.py -i input.jpg -o output.jpg --replace-black
"""
    
    args = parser.parse_args()
    
    # Determine which filter to apply
    filter_name = None
    if args.replace_black:
        filter_name = 'replace_black'
    
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
        filtered_image = apply_filter(image, filter_name)
        
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
