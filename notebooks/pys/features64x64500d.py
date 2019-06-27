from autoencoders import *
from pdb_utils import *

import pickle
import numpy as np
import pandas as pd
import random

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from functools import partial
import sys

from Bio import PDB
import numpy as np
import os
import cv2

import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Get and parse all pdb files in a folder
def parsePdbFiles(dir_path, sample_n=10):
    structures = []
    files = random.sample(os.listdir(dir_path), sample_n)
    print("Number of pdbs: %d"%len(files))
    pdb_files = [(f, os.path.join(dir_path, f)) for f in files if f.endswith(".pdb")]

    for pdb, pdb_path in pdb_files:
        parser = PDB.PDBParser()
        structure = parser.get_structure(pdb, pdb_path)
        structures.append(structure)
    return structures

with open('label_dict.pkl', 'rb') as f:
    label_dict = pickle.load(f)

def nice_time(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Runtime: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

print("Started parsing structures\n")
s_time = time.time()
structures = parsePdbFiles("PDBs")
end = time.time()
nice_time(s_time,end)


print("\nCreating protein contact maps")
s_time = time.time()
proteinmatrixdict = ProteinContactMapDict(structures, resize_to=(64,64), removeSymmetry=True)
end = time.time()
nice_time(s_time,end)

labels, features = list(proteinmatrixdict.keys()), list(proteinmatrixdict.values())
input_size = len(features[0])
print("Input size: ",input_size)

print("\nLinear Autoencoder - 100 epochs")
s_time = time.time()
new_features, loss = LinearAutoencoder(features, input_size, 500, 100, learning_rate=0.0001)
end = time.time()
nice_time(s_time,end)

# LOSS GRAPH
fig, ax = plt.subplots()
ax.plot(range(0,len(loss)), loss, 'go-', linewidth=1, markersize=1)
fig.savefig("loss_figure.png")

#new_feature_dict = dict(zip(labels,new_features))

y = []
for pdb_ in labels:
    y.append(label_dict[pdb_[:-4]])
print(y)

#print(y[20])
#print(labels[20])

print(len(y))
print(len(new_features))

# Stratified Cross Validation
from sklearn.model_selection import StratifiedShuffleSplit

print("Stratified Cross Validation - 10 splits\n")
s_time = time.time()

from sklearn.model_selection import cross_validate
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 42)
cv_results = cross_validate(classifier, new_features, y, cv=3)
print(cv_results)

"""
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
C = 1
acc, prec, rec = 0,0,0
for train_index, test_index in sss.split(new_features, y):
    print("Stratified Cross Validation - %d"%C)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("Random Forest Classifier - Training")
    # Fitting Random Forest Classification to the Training set
    classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 42)
    classifier.fit(X_train, y_train)
    # Predicting the Test set results
    print("Random Forest Classifier - Testing")
    y_pred = classifier.predict(X_test)
    acc+=accuracy_score(y_test, y_pred)
    prec+=precision_score(y_test, y_pred, average='weighted')
    rec+=recall_score(y_test, y_pred, average='weighted')
end = time.time()
nice_time(s_time,end)

print("\naverage accuracy_score: %f" %acc/10.0)
print("average precision_score: %f" %prec/10.0)
print("average recall_score: %f" %rec/10.0)
"""
