"""
BE noté : Classification de cibles radar à partir d'images ISAR (Radar à synthèse d'ouverture inverse.
By : LAHLOUH Jany
07/11/2021
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

####################################### Partie 1 #######################################
from dataLoader import load_bdd
from affichage import datasetAnalyse

path = "/home/jany/Documents/images_ISAR/"
images, labels = load_bdd(path)
datasetAnalyse(images, labels)


####################################### Partie 2 #######################################
import scipy.ndimage as sn
from calculs import polaire, polaire2




"Récupération de la première image du jeu de données puis transformation polaire"
# image=images[0][:,:,0]
image = cv2.imread(os.path.join("/home/jany/Documents/images_ISAR/", "A10_1.bmp"))

plt.figure("Transformation polaire")
plt.title("name")
plt.subplot(121)
plt.title("Image originelle")
plt.imshow(image)
plt.subplot(122)
image_pol = polaire(image)
plt.title("Image polaire")
plt.imshow(image_pol)





"Rotation de 45 degrés avec la bibliothèque scipy puis transformation polaire"

image45 = sn.rotate(image, 45)
image45polaire = polaire(image45)
image90 = sn.rotate(image, 90)
image90polaire = polaire(image90)


plt.figure("Images avec rotation")
plt.subplot(121)
plt.title("Image polaire tournée à 45°")
plt.imshow(image45polaire)
plt.subplot(122)
plt.title("Image polaire tournée à 90°")
plt.imshow(image90polaire)



"Changement d'échelle de l'image"
image256 = cv2.resize(image, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
image256polaire = polaire(image256)
image64 = cv2.resize(image, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
image64polaire = polaire(image64)

plt.figure("Changements d'echelle de l'image")
plt.subplot(121)
plt.title("Image 256 polaire")
plt.imshow(image256polaire)
plt.subplot(122)
plt.title("Image 64 polaire")
plt.imshow(image64polaire)
plt.show()




""" Seconde fonction de transformée polaire """
imagepolaire, ir, io = polaire2(image[:, :, 0])
image90polaire, ir2, io2 = polaire2(image90[:, :, 0])

plt.figure("Transformation polaire v2")
plt.subplot(131)
plt.title("image originelle")
plt.imshow(image)
plt.subplot(132)
plt.title("Image polaire")
plt.imshow(imagepolaire)
plt.subplot(133)
plt.title("Image à 90° polaire")
plt.imshow(image90polaire)



image2 = cv2.imread(os.path.join("/home/jany/Documents/images_ISAR/", "A10_140.bmp"))
image2polaire, ir, io = polaire2(image2[:, :, 0])
image2polaire90 =sn.rotate(image2polaire,90)

plt.figure("Transformation polaire v2 sur image 2")
plt.subplot(131)
plt.title("image originelle")
plt.imshow(image2)
plt.subplot(132)
plt.title("Image polaire")
plt.imshow(image2polaire)
plt.subplot(133)
plt.title("Image à 90° polaire")
plt.imshow(image2polaire90)
plt.show()

""" Inter corrélation """
image = cv2.imread(os.path.join("/home/jany/Documents/images_ISAR/", "A10_1.bmp"))
image = image[:, :, 0]
im, ir, io = polaire2(image)
# print(ir.shape)
import scipy.stats as ss

corr = ss.pearsonr(ir[:, 0], ir[:, 0])
#print(corr)
#cv2.imshow('corr', corr)
#cv2.waitKey(0)


####################################### Partie 3 #######################################
from sklearn.model_selection import train_test_split
import time
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score
from reconnaissance import labda_index, reconnaissance
from descripteursBDD import descripteurs, descripteurs2

descript = descripteurs2(images=images)
X_train, X_test, y_train, y_test = train_test_split(descript, labels, test_size=0.33, random_state=42)

print("Classifier C1")
C1 = reconnaissance(X_train, X_test, y_train, y_test, I='Ir', n_neighbors=1)
C1 = reconnaissance(X_train, X_test, y_train, y_test, I='Ir', n_neighbors=3)
C1 = reconnaissance(X_train, X_test, y_train, y_test, I='Ir', n_neighbors=5)
S1_X, S1_y = labda_index(X_train, y_train, X_test, taux=0.33, I='Ir')

print("Classifier C2")
C2 = reconnaissance(S1_X, X_test, S1_y, y_test, I='Io', n_neighbors=1)
C2 = reconnaissance(S1_X, X_test, S1_y, y_test, I='Io', n_neighbors=3)
C2 = reconnaissance(S1_X, X_test, S1_y, y_test, I='Io', n_neighbors=5)