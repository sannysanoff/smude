import sys
from skimage.io import imread, imsave
from smude import Smude

smude = Smude(use_gpu=False, binarize_output=True)

input_file = sys.argv[1] if len(sys.argv) > 1 else 'images/input_fullsize.jpg'
image = imread(input_file)
result = smude.process(image)
imsave('result.png', result)
