import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def keypoint_detector(img):
    
    sift = cv.SIFT_create()
    keypoints = sift.detect(img)
    
    return keypoints


def sift_descriptor(img, keypoints):
    
    sift = cv.SIFT_create()
    descriptors = sift.compute(img, keypoints)[1]

    return descriptors


def brisk_descriptor(img, keypoints):
    
    brisk = cv.BRISK_create()
    descriptors = brisk.compute(img, keypoints)[1]

    return descriptors


def sift_match(img1, img2):

    # keypoints and descriptors
    kp1, des1 = sift_descriptor(img1)
    kp2, des2 = sift_descriptor(img2)

    # create BFMatcher object
    BFMatcher = cv.BFMatcher()
    matches = BFMatcher.knnMatch(des1,des2,k=2)

    # Apply ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    # cv.drawMatchesKnn expects list of lists as matches
    output = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return output


def brisk_match(img1, img2):

    # keypoints and descriptors
    kp1, des1 = brisk_descriptor(img1)
    kp2, des2 = brisk_descriptor(img2)

    BFMatcher = cv.BFMatcher()

    matches = BFMatcher.match(queryDescriptors = des1, trainDescriptors = des2)
    matches = sorted(matches, key = lambda x: x.distance)

    # Draw first 15 matches
    output = cv.drawMatches(img1 = img1,
                            keypoints1 = kp1,
                            img2 = img2,
                            keypoints2 = kp2,
                            matches1to2 = matches[:15],
                            outImg = None,
                            flags = cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    return output


if __name__ == '__main__':
    img1 = cv.imread('../img/clean-bg/airplane/image_0230.jpg')
    img2 = cv.imread('../img/clean-bg/airplane/image_0231.jpg')

    img3 = brisk_match(img1, img2)
    # img3 = sift_match(img1, img2)

    plt.imshow(img3)
    plt.show()