import numpy as np
import os
import cv2

def loadFolder(folderPath, label, imagesList, labelsList):
    if type(folderPath) != str :
        print("folderPath must be str")
        exit()
    if type(label) != int :
        print("label must be int")
        exit()
    else:
        for filename in os.listdir(folderPath):
            img = cv2.cvtColor(cv2.imread(os.path.join(folderPath, filename)), cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (158,158))
            #if imagesList is None :
            #    imagesList = np.array(img)
            #else :
            #imagesList = np.hstack((imagesList, np.array(img)))
            imagesList.append(img)
            labelsList.append(label)
    return imagesList, labelsList


def loadDataset (DataPath):
    #print("Chargement des données du dossier", DataPath)
    imagesList = []
    labelsList = []
    for i in np.arange(1, 11):
        #print("Chargement de la classe", i)
        folderPath = DataPath + "/classe" + str(i)
        imagesList, labelsList = loadFolder(folderPath=folderPath, label=int(i)-1, imagesList=imagesList, labelsList=labelsList)
    return np.array(imagesList), np.array(labelsList)

