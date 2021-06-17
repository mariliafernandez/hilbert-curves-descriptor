# algorithm reference: http://blog.marcinchwedczuk.pl/iterative-algorithm-for-drawing-hilbert-curve

from PIL import Image, ImageDraw
from pathlib import Path, PurePath
import numpy as np

def index2xy(index, N, im_size=None):

    if im_size == None:
        height = N
        width = N
    else:
        width = im_size[0]
        height = im_size[1]

        
    # if N > im_size:
        # N = im_size
        
    # x, y positions in N=2
    positions = [
        [0,0],
        [0,1],
        [1,1],
        [1,0]
    ]
    
    # last 2 bits = position in N=2
    x, y = positions[ index&3 ]
    
    # next 2 bits = position in current N
    index = index >> 2
    
    n=4
    while n <= N:
        n2 = n//2
        
        h = index&3
        
        # Bottom left
        if h == 0:
            x, y = y, x
            
        # Upper left
        elif h == 1:
            x, y = x, y+n2
            
        # Upper right
        elif h == 2:
            x, y = x+n2, y+n2

        # Bottom right
        elif h == 3:
            x, y  = 2*n2-1-y, n2-1-x
            
        index = index >> 2
        n *= 2
    
    x, y = round(width/N*x), round(height/N*y)

    return x, y


def hilbert_order(N, im_size=None):
    img_curve = []

    for i in range(0, N*N):
        pixel_coord = index2xy(i, N, im_size)
        img_curve.append(pixel_coord)
    
    return np.asarray(img_curve)


def draw_curve_on_img(original_img, N):

    img = original_img.copy()
    curve = hilbert_order(N, img.size)

    draw = ImageDraw.Draw(img)
    draw.line(list(curve.flatten()), width=1, fill=128)

    return img


def pixel_values(img, curve_coordinates):

    curve_pixels = []
    img = np.asarray(img)

    for coord in curve_coordinates:
        curve_pixels.append(img[coord[0], coord[1]])
    
    return curve_pixels
    

def statistical_measures(pixel_curve):

    intensity_diff = []
    print(pixel_curve)
    for i in range(len(pixel_curve)):
        intensity_diff.append( pixel_curve[i+1] - pixel_curve[i] )

    print(intensity_diff)

    return intensity_diff


def descriptor(img, kp):
    curves = []
    n = 2

    while n <= min(img.size):
        curve_coordinates = hilbert_order(n, img.size)
        # draw = draw_curve_on_img(img, n)
        # draw.save(f'../img/hilbert/{n}.png')
        # print(np.sqrt(len(curve)))

        if kp in curve_coordinates:
            curve_pixels = pixel_values(img, curve_coordinates)
            curves.append(curve_pixels)
        n *= 2
    


import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('i', help="image")
    args = parser.parse_args()

    img = Image.open(args.i).convert('LA')
    
    descriptor(img, [10,43])