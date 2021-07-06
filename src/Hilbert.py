from PIL import Image, ImageDraw
from pathlib import Path, PurePath
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy
from descriptors import sift_descriptor, brisk_descriptor
import cv2 as cv

def pixel_values(self, img_array, curve_coordinates):

    curve_pixels = []

    for coord in curve_coordinates:
        curve_pixels.append(img_array[coord[1], coord[0]])
    
    return np.asarray(curve_pixels)
    
    
def histogram(self, array):
    
    hist = np.zeros(256)
            
    for i in range(256):
        hist[i] = int(np.count_nonzero(array == i))
        
    return np.asarray(hist)



class HilbertCurve:
    
    
    def index2xy(self, index, N, im_size):
        """
        Computes the coordinates in 2D (x,y) equivalent to a 1D Hilbert curve in level N
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


    def hilbert_order(self, N, im_size):
        """
        Returns a list of the coordinates of an image in a Hilbert Curve order
        """
        
        curve_coordinates = []

        for i in range(0, N*N):
            pixel_coord = self.index2xy(i, N, im_size)
            curve_coordinates.append(pixel_coord)
        
        return curve_coordinates


    def draw_curve_on_img(self, img_array, curve_coordinates):
        """
        Draw a Hilbert curve above an image
        """

        img = Image.fromarray(img_array)
        draw = ImageDraw.Draw(img)
        curve_array = np.asarray(curve_coordinates)
        # curve = np.asarray(hilbert_order(N, img.size))

        draw.line(list(curve_array.flatten()), width=1, fill=128)

        return img


    def compute(self, img_array, keypoints=None, max=None):
        """
        Computes de descriptors for a set of keypoints detected in an image
        """
        
        img = Image.fromarray(img_array)
        
        if max == None:
            max = len(keypoints)
            
        # Keypoints detection
        if keypoints == None:
            sift = cv.SIFT_create()
            keypoints = sift.detect(img_array)
        
        kp_index = 0
        desc_list = []
        curve_coordinates = self.hilbert_order(min(img.size), img.size)
        
        while kp_index < max:

            # Keypoint coordinates
            kp = keypoints[kp_index]
            (x, y) = (int(kp.pt[0]), int(kp.pt[1]))
            
            # Region of interest
            roi = self.roi_curve(img_array, curve_coordinates, kp)

            neighborhood = [
                (x-1,y), (x+1,y), (x,y-1), (x,y+1), 
                (x+1, y+1), (x+1, y-1), (x-1, y-1), (x-1, y+1)
                ]

            i=0
            while roi[0] is None and i < len(neighborhood):

                kp = neighborhood[i]
                roi = self.roi_curve(img_array, curve_coordinates, kp)
                i += 1
                
            # RoI descriptor
            if roi[0] is not None:
                desc_list.extend(self.roi_descriptor(roi[0]))
            
            kp_index += 1
        
        return np.asarray(desc_list, dtype='float32')
        
        
    def roi_descriptor(self, roi):
        """
        Computes de descriptor for a given Region of Interest in an image
        """
        
        desc_array = []
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
        
        desc_array.append(np.asarray([std1,std2,assi1,assi2,curt,entr]))
            
        return np.asarray(desc_array)
    

    def roi_coordinates(self, curve_coordinates, kp):
        """
        Returns a list of coordinates containing a Region of Interest around a keypoint
        """
        
        if kp in curve_coordinates:
        
            kp_index = curve_coordinates.index(kp)
            
            if kp_index > 64:
                return curve_coordinates[kp_index-64:kp_index+64]
            
            else:
                return curve_coordinates[:128]
        
        else: 
            return None
            
        
        
    def roi_curve(self, img_array, curve_coordinates, kp):
        """
        Returns an array of pixels of a Region of Interest around a keypoint of an image
        """
        
        roi = self.roi_coordinates(curve_coordinates, kp)
        
        if roi is None:
            return None, None
            
        else:
            roi_pixels = pixel_values(img_array, roi)
            return np.asarray(roi_pixels), np.asarray(roi) 