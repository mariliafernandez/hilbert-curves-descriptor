from descriptors import sift_descriptor, brisk_descriptor, keypoint_detector
from hilbert_curve import image_descriptor as hilbert_descriptor
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import cv2 as cv
import random
import time


descriptors = ['sift', 'brisk', 'hilbert']
classes = ['airplanes', 'motorbikes']

source_dir = Path('../caltech-101')
masks_path = source_dir / 'masks'
images_path = source_dir / 'segmented_images'

class_number = {}
i = 0

# Attribute number to class names
for classname in images_path.iterdir():
    if classname.name in classes:
        class_number[classname.name] = i
        i += 1


def compute_descriptors(n_keypoints):

    time_spent = {}
    label = {}
    data = {}

    for k in descriptors:
        time_spent[k] = []
        label[k] = []
        data[k] = []

    for classpath in images_path.iterdir():
    
        for f in list(classpath.iterdir()):

            target = class_number[classpath.name]

            # Read image
            img_array = cv.imread(str(f), cv.IMREAD_GRAYSCALE)
            
            # Read image mask
            mask_path = masks_path / classpath.name / f.name
            mask_array = cv.imread(str(mask_path), cv.IMREAD_GRAYSCALE)

            # Detect keypoints
            keypoints = keypoint_detector(img_array, mask_array)
            sorted_kp = sorted(keypoints, key = lambda x: x.size, reverse=True)     
            top_kp = sorted_kp[:n_kp]

            # Compute descriptors and time spent
            t1_sift = time.time()
            sift_desc = sift_descriptor(img_array, top_kp)
            t2_sift = time.time()

            t1_brisk = time.time()
            brisk_desc = brisk_descriptor(img_array, top_kp)
            t2_brisk = time.time()

            t1_hilbert = time.time()
            hilbert_desc = hilbert_descriptor(img_array, top_kp)
            t2_hilbert = time.time()

            # Save descriptor data
            data['sift'].extend(sift_desc)
            data['brisk'].extend(brisk_desc)
            data['hilbert'].extend(hilbert_desc)

            # Save image label
            label['sift'].extend(np.asarray([target]*len(sift_desc)))
            label['brisk'].extend(np.asarray([target]*len(brisk_desc)))
            label['hilbert'].extend(np.asarray([target]*len(hilbert_desc)))

            # Save time spent
            time_spent['sift'].append(t2_sift - t1_sift)
            time_spent['brisk'].append(t2_brisk - t1_brisk)
            time_spent['hilbert'].append(t2_hilbert - t1_hilbert)

    for k in descriptors:
        label[k] = np.asarray(label[k])
        time_spent[k] = np.asarray(time_spent[k])

    return data, label, time_spent


def balance_set(data, label):
    
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(data, label)
    
    return X_resampled, y_resampled


def create_model_report(classifier, X, y):
    
    data, label = balance_set(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(data, label)
    
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    report = classification_report(y_test, predicted, output_dict=True, target_names=['motorbike', 'airplane'])
    
    model_report_all = {'sift':{}, 'brisk':{}, 'hilbert':{}}

    for key in descriptors:
        model_report_all[key]['keypoints'] = n
        model_report_all[key]['accuracy'] = np.round(report[key]['accuracy'], 2)

        for metric in ['precision', 'recall', 'f1-score']:
            model_report_all[key][metric] = np.round(report[key]['weighted avg'][metric], 2)
        
        model_report_all[key]['tempo'] = np.round(np.average(time_spent[key]), 4)
        
    return report
    

if __name__ == '__main__':

    knn_report = {}

    for n in [50, 100, 150]:
        data, label, time_spent = compute_descriptors(n)

        for key in descriptors:
            knn = KNeighborsClassifier(n_neighbors=3)
            knn_report[key] = create_model_report(knn, data[key], label[key])