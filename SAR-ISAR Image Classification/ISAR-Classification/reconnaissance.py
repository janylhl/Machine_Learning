import time
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

def reconnaissance(X_train, X_test, y_train, y_test, I, n_neighbors):
    C = KNeighborsClassifier(n_neighbors)
    if I == 'Ir':
        start = time.time()
        C.fit(X_train[:,0:49], y_train)
        end = time.time()
        temps = end - start
        y_pred = C.predict(X_test[:,0:49])
    elif I == 'Io':
        start = time.time()
        C.fit(X_train[:, 50:99], y_train)
        end = time.time()
        temps = end - start
        y_pred = C.predict(X_test[:, 50:99])
    else :
        print("I value incorrect, should be Ir or Io")
        exit()
    print("Temps d'entrainement : ", temps)
    print("Classification report pour K =  ", n_neighbors)
    print(classification_report(y_test, y_pred))
    print(type(classification_report(y_test, y_pred)))

    return C

def labda_index(X_train, y_train, X_test, I, taux=0.33, n_neighbors=1):
    C1=NearestNeighbors(n_neighbors)
    if I == 'Ir':
        C1.fit(X_train[:,0:49], y_train)
        distances, indices = C1.kneighbors(X_test[:,0:49], return_distance=True)
    elif I == 'Io':
        C1.fit(X_train[:, 50:49], y_train)
        distances, indices = C1.kneighbors(X_test[:, 50:99], return_distance=True)
    else :
        print("I value incorrect, should be Ir or Io")
        exit()
    S1_idx = indices[0:int(taux+len(indices))]
    S1_X = []
    S1_y = []
    for idx, img in enumerate(X_train):
        if idx in S1_idx:
            S1_X.append(img)
            S1_y.append(y_train[idx])
    return np.array(S1_X), np.array(S1_y)