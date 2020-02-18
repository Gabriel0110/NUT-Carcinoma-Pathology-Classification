import image_augmentation
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import random
import skimage as sk
import skimage.io as skio
import os
import numpy as np
import time

def main():
    data = []
    
    nmc = 'C:\\Users\\gabri\\Desktop\\NMC Pathology\\nmc'
    non_nmc = 'C:\\Users\\gabri\\Desktop\\NMC Pathology\\non-nmc'

    nmc_images = [os.path.join(nmc, f) for f in os.listdir(nmc) if os.path.isfile(os.path.join(nmc, f))]
    non_nmc_images = [os.path.join(non_nmc, f) for f in os.listdir(non_nmc) if os.path.isfile(os.path.join(non_nmc, f))]
    
    nmc_labels = [1 for _ in range(len(nmc_images))]
    non_nmc_labels = [0 for _ in range(len(non_nmc_images))]
    
    targets = nmc_labels + non_nmc_labels
    
    nmc_imgs = []
    non_nmc_imgs = []
    
    for pic in nmc_images:
        img = skio.imread(pic)
        img = sk.transform.resize(img, (150, 150), anti_aliasing=True)
        #img = sk.color.rgb2gray(img)
        nmc_imgs.append(img.flatten())
        
    for pic in non_nmc_images:
        img = skio.imread(pic)
        img = sk.transform.resize(img, (150, 150), anti_aliasing=True)
        #img = sk.color.rgb2gray(img)
        non_nmc_imgs.append(img.flatten())
    
    data = nmc_imgs + non_nmc_imgs
    data = np.asarray(data)
    targets = np.asarray(targets)
    
    x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.3, random_state=0)
    
    svc = svm.SVC(random_state=0)
    xgb = XGBClassifier(random_state=0)
    dtc = DecisionTreeClassifier(random_state=0)
    nbc = GaussianNB()
    rfc = RandomForestClassifier(random_state=0)
    gbc = GradientBoostingClassifier(random_state=0)
    mlp = MLPClassifier(random_state=0)
    
    models = {"SVC": svc, "XGBoost": xgb, "DecisionTree": dtc, "NaiveBayes": nbc, "RandomForest": rfc, "GradientBoost": gbc, "NeuralNet": mlp}
    params = {}
    
    for name, model in models.items():
        if model == svc:
            params = {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 'kernel': ['rbf', 'linear']}
        elif model == xgb:
            params = {}
        elif model == dtc:
            params = {'max_depth': [1, 2, 3, 4]}
        elif model == nbc:
            params = {}
        elif model == mlp:
            params = {'hidden_layer_sizes': [[128, 64, 32], [32, 32, 32]], 'activation': ['tanh', 'relu'], 'solver': ['sgd', 'adam'], 'alpha': [0.0001, 0.001, 0.01, 0.1], 'max_iter': [200, 300, 400]}
        elif model == rfc:
            params = {'n_estimators': [5, 10, 16, 32, 64, 128]}
        elif model == gbc:
            params = {'n_estimators': [5, 10, 16, 32, 64, 128]}
        
        start_time = time.time()
        
        clf = GridSearchCV(model, params, cv=5)
        clf.fit(x_train, y_train)
        pred = clf.predict(x_test)
        confusion = confusion_matrix(y_test, pred)
        
        duration = time.time() - start_time
        
        print("\n\n--- {} ---".format(name))
        print("Total Duration: {}".format(duration))
        print("Score: {}".format(clf.score(x_test, y_test)))
        print("Confusion Matrix:")
        print("{}".format(confusion))
    
if __name__ == '__main__':
    main()
