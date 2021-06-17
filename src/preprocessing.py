from PIL import Image
import numpy as np
from pathlib import Path

def crop_square(img_original):
    
    img = img_original.copy()
    size = min(img.size)
    print(img.size)

    bit_size = size.bit_length()
    crop_size = np.power(2, bit_size-1)

    left = img.size[0]//2 - crop_size//2
    right = img.size[0]//2 + crop_size//2
    upper = img.size[1]//2 - crop_size//2
    lower = img.size[1]//2 + crop_size//2

    img_crop = img.crop((left, upper, right, lower))
        
    print(left, upper, right, lower)
    img_crop.save(Path(f'../img/hilbert/crop_test.png'))


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('i', help="image")
    args = parser.parse_args()

    with Image.open(args.i) as f:
        img = f.copy()
    
    crop_square(img)
