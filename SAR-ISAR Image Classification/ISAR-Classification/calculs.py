import cv2
import numpy as np
from skimage.transform import warp_polar
from sklearn.neighbors import NearestNeighbors

def polaire(img):
    img2 = cv2.linearPolar(img, (img.shape[0] / 2, img.shape[1] / 2), 40, cv2.WARP_FILL_OUTLIERS)
    return img2


def polaire2(img):
    img2 = warp_polar(img, radius=64, output_shape=(50, 50))
    io = np.zeros((np.shape(img2)[0], 1))
    ir = np.zeros((np.shape(img2)[1], 1))
    for r in range(np.shape(img2)[0]):
        io[r, 0] = np.sum(img2[r, :])
    for o in range(np.shape(img2)[1]):
        ir[o, 0] = np.sum(img2[:, o])
    return img2, ir, io

