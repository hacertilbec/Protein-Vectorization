from pdb_utils import *

import pickle
import numpy as np
import pandas as pd
import random

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from functools import partial
import sys

from Bio import PDB
import os
import cv2

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import time

def nice_time(start,end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Runtime: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))

def createDistanceMatrix2(structure,mean,std_dev,resize_strategy,resize_to,sample_size):
    coords_list = []
    model=structure[0]
    for chain in model.get_list():
        for residue in chain.get_list():
            try:
                coords = residue['CA'].coord
                coords_list.append(coords)
            except:
                continue
    distance_matrix = []
    for c in coords_list:
        coord_dist = []
        for c_ in coords_list:
            dist = np.linalg.norm(c-c_) # calculate distance between coordinates of CAs of residues
            coord_dist.append(dist)
        distance_matrix.append(coord_dist)
    print(np.array(distance_matrix))
    distance_matrix = (np.array(distance_matrix)-mean)/std_dev
    print(distance_matrix)
    # Resize protein_matrix
    if resize_strategy == False or len(distance_matrix) == resize_to[0]:
        return distance_matrix
    else:
        if resize_strategy == "strategy1":
            resized = cv2.resize(distance_matrix, (resize_to[0], resize_to[1]), interpolation=cv2.INTER_AREA)
        elif resize_strategy == "strategy2":
            if len(distance_matrix) > resize_to[0]:
                resized = sampling(distance_matrix, new_shape=resize_to,sample_size=sample_size)
            else:
                resized = padding(distance_matrix, new_shape=resize_to,sample_size=sample_size)
        elif resize_strategy == "strategy3":
            if len(distance_matrix) > resize_to[0]:
                resized = sampling_s3(distance_matrix, new_shape=resize_to)
            else:
                resized = padding_s3(distance_matrix, new_shape=resize_to)
        else:
            print("Not a valid strategy method. Use False, strategy1, strategy2 or strategy3.")
            return
    return resized


# get structure list and returns protein, distance matrix dictionary
def DistanceMatrixDict2(structures,mean,std_dev,resize_strategy="strategy1", resize_to=(32,32), removeSymmetry=False,sample_size=None):
    if resize_strategy == "strategy2" and removeSymmetry == True:
        print("RemoveSymmetry parameter can not be used with strategy2")
        return
    protein_matrix_dict = {}
    for protein in structures:
        protein_matrix = createDistanceMatrix2(protein,mean,std_dev,resize_strategy, resize_to,sample_size)
        if resize_strategy == "strategy2" or resize_strategy == "strategy3":
            try:
                protein_matrix[0].shape[1]
                for sample in enumerate(protein_matrix):
                    key = protein.id + "sample" + str(sample[0])
                    protein_matrix_dict[key] = sample[1].flatten()
            except:
                protein_matrix_dict[protein.id] = protein_matrix.flatten()
        else:
            if type(protein_matrix) == np.ndarray:
                if removeSymmetry == True:
                    protein_matrix = RemoveSymmetry(protein_matrix)
                else:
                    protein_matrix = protein_matrix.flatten()
                protein_matrix_dict[protein.id] = protein_matrix
    return protein_matrix_dict

# Loading label dictionaries
with open('pickle files/fold_groups.pkl', 'rb') as f:
    fold_dict = pickle.load(f)
with open('pickle files/label_dict.pkl', 'rb') as f:
    label_dict = pickle.load(f)
with open('pickle files/test_labels.pkl', 'rb') as f:
    test_label_dict = pickle.load(f)


tf.reset_default_graph()
tf.set_random_seed(42)

encoding_size = 50
filter_size = 64

train_all_features = []
train_all_pdb_names = []

X = tf.placeholder(tf.float32, shape=[None, 2016])
hidden = fully_connected(X, 50, activation_fn=tf.nn.relu)
outputs = fully_connected(hidden, 2016, activation_fn=tf.nn.relu)

reconstruction_loss = tf.reduce_sum(tf.square(outputs - X)) # MSE

optimizer = tf.train.AdamOptimizer(0.0001)
training_op = optimizer.minimize(reconstruction_loss)

codings = hidden # the output of the hidden layer provides the codings

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(init)

all_values = []
for fold,pdb_list in list(fold_dict.items())[:10]:
    if len(pdb_list)<2:
        continue
    for pdb in pdb_list:
        try:                
            pdb_path = os.path.join("PDBs", pdb+".pdb")
            parser = PDB.PDBParser()
            structure = parser.get_structure(pdb, pdb_path)
            coords_list = []
            model=structure[0]
            for chain in model.get_list():
                for residue in chain.get_list():
                    try:
                        coords = residue['CA'].coord
                        coords_list.append(coords)
                    except:
                        continue
            distance_matrix = []
            for c in coords_list:
                coord_dist = []
                for c_ in coords_list:
                    dist = np.linalg.norm(c-c_) # calculate distance between coordinates of CAs of residues
                    coord_dist.append(dist)
                distance_matrix.append(coord_dist)
            flatten = []
            row = 0
            for i in range(1, len(distance_matrix)):
                flatten += distance_matrix[row][i:]
                row+=1
            all_values+=flatten
        except:
            continue
            
all_values = np.array(all_values)
mean = np.mean(all_values)
std_dev = np.std(all_values)

# Distance Matrices
train_all_pdb_names, train_all_features = [],[]

for fold,pdb_list in list(fold_dict.items())[:10]:
    if len(pdb_list)<2:
        continue
    structures = []
    for pdb in pdb_list:
        try:                
            pdb_path = os.path.join("PDBs", pdb+".pdb")
            parser = PDB.PDBParser()
            structure = parser.get_structure(pdb, pdb_path)
            structures.append(structure)
            train_matrix = DistanceMatrixDict2(structures, mean, std_dev, resize_strategy="strategy1", resize_to=(filter_size,filter_size),removeSymmetry=True)
            train_pdb_names, train_features = list(train_matrix.keys()), list(train_matrix.values())
            input_size = len(train_features[0])

            train_all_pdb_names += train_pdb_names
            train_all_features += train_features

        except:
            continue
X_train, X_test, y_train, y_test = train_test_split(train_all_features, train_all_pdb_names, test_size=0.2,random_state=42)
print("Train size: %d"%len(X_train))
print("Test size: %d"%len(X_test))
# Autoencoder
for iteration in range(100):
    print("Iteration %d"%iteration)
    start,end = 0, 500
    for i in range(int(len(X_train)/500.)+1):
        batch = X_train[start:end]
        # Train model with the batch
        sess.run([training_op], feed_dict={X: np.array(batch)})

        start+=500
        end+=500
 

print("Training step end.")
curdir = os.getcwd()
# Save the variables to disk.
save_path = saver.save(sess, curdir+"/models/model.ckpt")
print("Model saved in path: %s" % save_path)

# Encode Train pdbs
new_train_features = codings.eval(feed_dict={X: np.array(X_train)},session=sess)
# Encode Test pdbs
new_test_features = codings.eval(feed_dict={X: np.array(X_test)},session=sess)

# Prepare Scop Test data
scop_structures = []
for pdb in os.listdir("SCOP_Test/"):
    pdb_path = os.path.join("SCOP_Test", pdb)
    parser = PDB.PDBParser()
    structure = parser.get_structure(pdb, pdb_path)
    scop_structures.append(structure)
scop_matrix = DistanceMatrixDict(scop_structures, resize_strategy="strategy1", resize_to=(filter_size,filter_size),removeSymmetry=True)
y_scop, scop_features = list(scop_matrix.keys()), list(scop_matrix.values())

# Encode Scop Test pdbs
scop_features = codings.eval(feed_dict={X: scop_features},session=sess)



# Prepare train X and y
train_feature_dict = {}
for i in enumerate(y_train):
    if "sample" in i[1]:
        pdb = i[1].split("sample")[0]
    else:
        pdb = i[1]
    train_feature_dict.setdefault(pdb,[])
    train_feature_dict[pdb].append(new_train_features[i[0]])

# Preparing test X and y
test_feature_dict = {}
for i in enumerate(y_test):
    if "sample" in i[1]:
        pdb = i[1].split("sample")[0]
    else:
        pdb = i[1]
    test_feature_dict.setdefault(pdb,[])
    test_feature_dict[pdb].append(new_test_features[i[0]])

# Preparing Scop test X and y
scop_feature_dict = {}
for i in enumerate(y_scop):
    if "sample" in i[1]:
        pdb = i[1].split("sample")[0]
    else:
        pdb = i[1]
    scop_feature_dict.setdefault(pdb,[])
    scop_feature_dict[pdb].append(scop_features[i[0]])

X_train = []
y_train = []

for pdb,vector in train_feature_dict.items():
    X_train.append(np.average(vector,axis=0))
    y_train.append(".".join(label_dict[pdb].split(".")[:2]))

X_test = []
y_test = []

for pdb,vector in test_feature_dict.items():
    X_test.append(np.average(vector,axis=0))
    y_test.append(".".join(label_dict[pdb.split(".")[0]].split(".")[:2]))

X_scop = []
y_scop = []

for pdb,vector in scop_feature_dict.items():
    X_scop.append(np.average(vector,axis=0))
    y_scop.append(".".join(test_label_dict[pdb.split(".")[0]].split(".")[:2]))


uniques = list(set(y_train).union(set(y_test)).union(set(y_scop)))
group2id = dict(zip(uniques, range(len(uniques))))

X_train = np.array(X_train)
y_train = np.array(list(map(lambda x: group2id[x], y_train)))
X_test = np.array(X_test)
y_test = np.array(list(map(lambda x: group2id[x], y_test)))
X_scop = np.array(X_scop)
y_scop = np.array(list(map(lambda x: group2id[x], y_scop)))


print("Random Forest starts")
sss1 = StratifiedShuffleSplit(n_splits=2, test_size=0.25, random_state=42)
for a, b in sss1.split(X_train, y_train):
    X_train_, y_train_ = X_train[a], y_train[a]
    X_validation, y_validation = X_train[b], y_train[b]

    # Hyperparameter Optimization with validation set
    params = {'max_depth':[3,4,5,6,7,8,9,10,15,20],
          'criterion':('gini', 'entropy'),
          'warm_start':(True,False),
         'n_estimators': (10,50,100,200,500)}

    rf = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(rf, params, cv=5, refit=True)
    clf.fit(X_validation, y_validation)

    # Training best model with train and validation set
    model = clf.best_estimator_
    model.fit(X_train, y_train)

    # Train and Test Accuracy Scores
    train_acc = model.score(X_train,y_train)
    test_acc = model.score(X_test,y_test)
    scop_acc = model.score(X_scop,y_scop)
    print(clf.best_params_)
    print((train_acc,test_acc,scop_acc))
