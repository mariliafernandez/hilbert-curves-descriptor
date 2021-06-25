# algorithm reference: http://blog.marcinchwedczuk.pl/iterative-algorithm-for-drawing-hilbert-curve

from PIL import Image, ImageDraw
from pathlib import Path, PurePath
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy
from descriptors import sift_descriptor, brisk_descriptor
import cv2 as cv

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
    
    return img_curve


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
        curve_pixels.append(img[coord[1], coord[0]])
    
    return np.asarray(curve_pixels)
    

def statistical_measures(pixel_curve):

    intensity_diff = []

    for i in range(len(pixel_curve)):
        intensity_diff.append( int(pixel_curve[i+1]) - int(pixel_curve[i]) )

    return np.asarray(intensity_diff)


def histogram(array):
    
    hist = np.zeros(256)
            
    for i in range(256):
        hist[i] = int(np.count_nonzero(array == i))
        
    return np.asarray(hist)


def keypoint_descriptor(img, kp):
    
    desc_array = []
    n = 2

    while n <= min(img.size):
        
        curve_coordinates = hilbert_order(n, img.size)

        try:
            kp_index = curve_coordinates.index(kp)
        except ValueError:
            # print(kp)
            n *= 2
            continue
        
        roi = curve_coordinates[kp_index-32:kp_index+32]
        roi_pixels = pixel_values(img, roi)
        
        hist = histogram(roi_pixels) # Histogram of intensity values
        prob_dist = hist/np.sum(hist)  # Probability distribution
        
        grad = np.gradient(roi_pixels)
        
        # Standard Deviation
        std1 = np.std(roi_pixels)
        std2 = np.std(grad)
        
        # Assimetry
        assi1 = skew(roi_pixels)
        assi2 = skew(grad)
        
        # Kurtosis
        curt = kurtosis(prob_dist)
        
        # Entropy
        entr = entropy(prob_dist)
        
        desc_array.append(np.asarray([std1,std2,assi1,assi2,curt,entr]))
        # print(desc_array)
            
        n *= 2
        
    return np.asarray(desc_array)
    

def image_descriptor(img, keypoints=None):
    
    img_desc = []
    
    if keypoints == None:
        sift = cv.SIFT_create()
        keypoints = sift.detect(np.asarray(img))
    
    for kp in keypoints:
        x, y = kp.pt
        img_desc.append(keypoint_descriptor(img, (round(x), round(y))))
    
    return np.asarray(img_desc)
    
    
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('i', help="image")
    args = parser.parse_args()

    img = Image.open(args.i).convert('L')
    
    img_desc = image_descriptor(img)
    
    print(img_desc)