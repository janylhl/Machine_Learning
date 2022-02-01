import numpy as np
from calculs import polaire2


def descripteurs(images, I):
    for idx, img in enumerate(images):
        img = img[:, :, 0]
        im, ir, io = polaire2(img)
        if I == 'Ir':
            Ivect = np.vstack((ir)).T
        elif I == 'Io':
            Ivect = np.vstack((io)).T
        elif I == 'Iro':
            Ivect = np.vstack((ir)).T + np.vstack((io)).T
        else:
            print("I value not correct, should be Ir or Io")
            break
        if idx == 0:
            matrix_I = Ivect
        if idx != 0:
            matrix_I = np.concatenate((matrix_I, Ivect))
    return matrix_I

def descripteurs2(images):
    for idx, img in enumerate(images):
        img = img[:, :, 0]
        im, ir, io = polaire2(img)
        Ivect = np.hstack((np.vstack((ir)).T, np.vstack((io)).T))
        if idx == 0:
            matrix_I = Ivect
        if idx != 0:
            matrix_I = np.concatenate((matrix_I, Ivect))
    return matrix_I

