#!/usr/bin/env python3
"""
Image Filter Tool
Applies various filters to images with command line interface.
"""

import argparse
import sys
import cv2
import numpy as np


def replace_black_filter(image):
    """Replace absolute black pixels (0) with white (255)."""
    # Create mask for pixels that are exactly 0
    mask = (image == 0)
    
    # Replace 0 values with 255
    result = image.copy()
    result[mask] = 255
    
    return result


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
                              help='Replace absolute black pixels (0) with white (255)')
    
    # Add help for filters
    parser.epilog = """
Available Filters:
  --replace-black    Replace absolute black pixels (0) with white (255)
                     No additional parameters required.

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
