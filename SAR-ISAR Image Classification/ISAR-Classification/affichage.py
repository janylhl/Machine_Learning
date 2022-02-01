import numpy as np

def datasetAnalyse(images, labels):
    print("----------Analyse du datadet----------")
    print()
    print("Nombre total d'éléments = ", len(images))
    print("Nombre de classes : 5")
    print("Nombre d'éléments labélisés A10 : =", labels.count(0))
    print("Nombre d'éléments labélisés F4 : = ", labels.count(1))
    print("Nombre d'éléments labélisés F14 : =", labels.count(2))
    print("Nombre d'éléments labélisés F15:  =", labels.count(3))
    print("Nombre d'éléments labélisés F16 : =", labels.count(4))
    print("Valeur minimale atteinte : ", np.min(images))
    print("Valeur maximale atteinte : ", np.max(images))
    print()
    print("--------------------------------------")
    print()