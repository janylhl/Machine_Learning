import cv2
import os
import re


def load_bdd(path):
    images = []
    classes = []
    for filename in os.listdir(path):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img[64:192, 64:192])
        if re.search('A10', filename):
            classes.append(0)
        if re.search('F4', filename):
            classes.append(1)
        if re.search('F14', filename):
            classes.append(2)
        if re.search('F15', filename):
            classes.append(3)
        if re.search('F16', filename):
            classes.append(4)
    return images, classes
