# algorithm reference: http://blog.marcinchwedczuk.pl/iterative-algorithm-for-drawing-hilbert-curve

from PIL import Image, ImageDraw
from pathlib import Path, PurePath
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy
from descriptors import sift_descriptor, brisk_descriptor
import cv2 as cv


def index2xy(index, N:int, im_size):
    """
    Transforms index of hilbert curve to x,y coordinates
    """
    width = im_size[0]
    height = im_size[1]
        
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
    
    x, y = int(width/(n//2)*x), int(height/(n//2)*y)

    return x, y


def hilbert_order(N:int, im_size):
    """
    Returns a list of coordinates in a Hilbert curve order
    """
    curve_coordinates = []

    for i in range(0, N*N):
        pixel_coord = index2xy(i, N, im_size)
        curve_coordinates.append(pixel_coord)
    
    return curve_coordinates


def draw_curve_on_img(img_array:np.ndarray, curve_coordinates):
    """
    Returns an image with its Hilbert curve drawing
    """
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    curve_array = np.asarray(curve_coordinates)
    # curve = np.asarray(hilbert_order(N, img.size))

    draw.line(list(curve_array.flatten()), width=1, fill=128)

    return img


def pixel_values(img_array:np.ndarray, curve_coordinates):
    """
    Returns a numpy array containing the pixel values 
    in the img_array, in the given order in curve_coordinates
    """
    curve_pixels = []

    for coord in curve_coordinates:
        curve_pixels.append(img_array[coord[1], coord[0]])
    
    return np.asarray(curve_pixels)


def histogram(array):
    
    hist = np.zeros(256)
            
    for i in range(256):
        hist[i] = int(np.count_nonzero(array == i))
        
    return np.asarray(hist)


def roi_descriptor(roi):
    
    n = 2

    hist = histogram(roi) # Histogram of intensity values
    prob_dist = hist/np.sum(hist)  # Probability distribution
    
    grad = np.gradient(roi)
    
    # Standard Deviation
    std1 = np.std(roi)
    std2 = np.std(grad)
    
    # Assimetry
    assi1 = skew(roi)
    assi2 = skew(grad)
    
    # Kurtosis
    curt = kurtosis(prob_dist)
    
    # Entropy
    entr = entropy(prob_dist)
    
    desc_array = np.asarray([std1,std2,assi1,assi2,curt,entr])
        
    return desc_array
    

def image_descriptor(img_array:np.ndarray, keypoints):
    
    img = Image.fromarray(img_array).convert("L")
    
    kp_index = 0
    desc_list = []
    curve_coordinates = hilbert_order(min(img.size), img.size)
    
    while kp_index < len(keypoints):

        # Keypoint coordinates
        kp = keypoints[kp_index]
        (x, y) = (int(kp.pt[0]), int(kp.pt[1]))
        
        # Region of interest
        roi = roi_curve(img_array, curve_coordinates, kp)

        # If keypoint not in the curve, look for kp in the neighbothood
        neighborhood = [
            (x-1,y), (x+1,y), (x,y-1), (x,y+1), 
            (x+1, y+1), (x+1, y-1), (x-1, y-1), (x-1, y+1)
            ]

        i=0
        while roi[0] is None and i < len(neighborhood):

            kp = neighborhood[i]
            roi = roi_curve(img_array, curve_coordinates, kp)
            i += 1
            
        # RoI descriptor
        if roi[0] is not None:
            desc_list.append(roi_descriptor(roi[0]))
        
        kp_index += 1
    
    return np.asarray(desc_list)
    

def roi_coordinates(curve_coordinates, kp):
    
    if kp in curve_coordinates:
    
        kp_index = curve_coordinates.index(kp)
        
        if kp_index > 64:
            return curve_coordinates[kp_index-64:kp_index+64]
        
        else:
            return curve_coordinates[:128]
    
    else: 
        return None
        
    
    
def roi_curve(img_array:np.ndarray, curve_coordinates, kp):
    
    roi = roi_coordinates(curve_coordinates, kp)
    
    if roi is None:
        return None, None
        
    else:
        roi_pixels = pixel_values(img_array, roi)
        return np.asarray(roi_pixels), np.asarray(roi)

    
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('i', help="image")
    args = parser.parse_args()

    img = Image.open(args.i).convert('L')
    
    img_desc = image_descriptor(img)
    
    print(img_desc)