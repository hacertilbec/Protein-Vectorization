#from autoencoders import *
from pdb_utils import *
from autoencoders import *
import pickle
import numpy as np
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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import *

def top5score(index_dict,neighbors,testY):
    score = 0
    for i,nb in enumerate(neighbors):
        y = testY[i]
        for n in nb:
            if index_dict[n]==y:
                score+=1
                break
    return(score/float(len(neighbors)))

filter_size=64
strategy = "strategy1"
encoding_size = 50
batch_size = 100
num_iters = 100
input_size=2016

# LOAD LABEL DICTIONARIES
with open('pickle files/label_dict.pkl', 'rb') as f:
    label_dict = pickle.load(f)
with open('pickle files/test_labels.pkl', 'rb') as f:
    scop_label_dict = pickle.load(f)

with open("pickle files/fold_groups.pkl","r") as f:
    fold_groups = pickle.load(f)
with open("scop_fold_groups.pkl","r") as f:
    scop_fold_groups = pickle.load(f)

selected = ['a.5','c.67','c.1','b.36']

X_train_pdbs = []
for fold in fold_groups:
    if fold in selected:
        X_train_pdbs+=fold_groups[fold]
print(len(X_train_pdbs))

X_test_pdbs = []
for fold in scop_fold_groups:
    if fold in selected:
        X_test_pdbs+=scop_fold_groups[fold]
print(len(X_test_pdbs))


def scale_features(features,mean,std_dev):
    return (np.array(features)-mean)/std_dev


structures = []
for pdb in X_test_pdbs:
    pdb=pdb+".pdb"
    pdb_path = os.path.join("SCOP_Test", pdb)
    parser = PDB.PDBParser()
    structure = parser.get_structure(pdb, pdb_path)
    structures.append(structure)

matrixdict = DistanceMatrixDict(structures, resize_strategy=strategy, resize_to=(filter_size,filter_size),removeSymmetry=True)
scop_pdb_samples, scop_features = list(matrixdict.keys()), list(matrixdict.values())




structures = []
for pdb in X_train_pdbs:
    pdb=pdb+".pdb"
    pdb_path = os.path.join("PDBs", pdb)
    parser = PDB.PDBParser()
    structure = parser.get_structure(pdb, pdb_path)
    structures.append(structure)

matrixdict = DistanceMatrixDict(structures, resize_strategy=strategy, resize_to=(filter_size,filter_size),removeSymmetry=True)
train_pdb_samples, train_features = list(matrixdict.keys()), list(matrixdict.values())

input_size = len(train_features[0])

# SCALING
train_features = np.array(train_features)
mean = np.mean(train_features.flatten())
std_dev = np.std(train_features.flatten())
train_features = scale_features(train_features,mean,std_dev)

# AUTOENCODER
tf.reset_default_graph()
tf.set_random_seed(42)

X = tf.placeholder(tf.float32, shape=[None, input_size])
hidden = fully_connected(X, encoding_size, activation_fn=tf.nn.elu)
outputs = fully_connected(hidden, input_size, activation_fn=None)

reconstruction_loss = tf.reduce_sum(tf.square(outputs - X)) # MSE

optimizer = tf.train.AdamOptimizer(0.0001)
training_op = optimizer.minimize(reconstruction_loss)

codings = hidden # the output of the hidden layer provides the codings

init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)

# AUTOENCODER TRAINING
losses = []
for iteration in range(num_iters):
    batch_loss = []
    for i in range(int(len(train_features)/batch_size)+1): # BATCH
        batch = train_features[i*batch_size: (i+1)*batch_size]
        _,loss = sess.run([training_op,reconstruction_loss], feed_dict={X: batch})
        batch_loss.append(loss)
    losses.append(sum(batch_loss)/float(len(batch_loss)))
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
plt.plot(range(num_iters),losses)
plt.savefig("loss_plot2.png")

# SCALING
scop_features = scale_features(scop_features,mean,std_dev)

# REPORT
print("number of train pdbs: {n}".format(n=len(train_pdb_samples)))

# Embedding vectors of train and test set
new_train_features = sess.run([hidden], feed_dict={X: train_features})[0]
new_scop_features = sess.run([hidden], feed_dict={X: scop_features})[0]

train_feature_dict = {}
for i in enumerate(train_pdb_samples):
    if "sample" in i[1]:
        pdb = i[1].split("sample")[0]
    else:
        pdb = i[1]
    train_feature_dict.setdefault(pdb,[])
    train_feature_dict[pdb].append(new_train_features[i[0]])

scop_feature_dict = {}
for i in enumerate(scop_pdb_samples):
    if "sample" in i[1]:
        pdb = i[1].split("sample")[0]
    else:
        pdb = i[1]
    scop_feature_dict.setdefault(pdb,[])
    scop_feature_dict[pdb].append(new_scop_features[i[0]])


X_train = []
y_train = []

for pdb,vector in train_feature_dict.items():
    pdb = pdb.split(".")[0]
    X_train.append(np.average(vector,axis=0))
    y_train.append(".".join(label_dict[pdb].split(".")[:2]))


X_scop = []
y_scop = []

for pdb,vector in scop_feature_dict.items():
    pdb = pdb.split(".")[0]
    X_scop.append(np.average(vector,axis=0))
    y_scop.append(".".join(scop_label_dict[pdb.split(".")[0]].split(".")[:2]))


uniques = list(set(y_train).union(set(y_scop)))
group2id = dict(zip(uniques, range(len(uniques))))

X_train = np.array(X_train)
y_train = np.array(list(map(lambda x: group2id[x], y_train)))

X_scop = np.array(X_scop)
y_scop = np.array(list(map(lambda x: group2id[x], y_scop)))

index_train = {}
for i,train in enumerate(y_train):
    index_train[i] = train

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(X_train)
X_train = pca.transform(X_train)
X_scop = pca.transform(X_scop)
# CLASSIFICATION

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
print(top5score(index_train,knn.kneighbors(X_train,n_neighbors=1,return_distance=False),y_train))
print(top5score(index_train,knn.kneighbors(X_scop,n_neighbors=1,return_distance=False),y_scop))

print(top5score(index_train,knn.kneighbors(X_train,n_neighbors=5,return_distance=False),y_train))
print(top5score(index_train,knn.kneighbors(X_scop,n_neighbors=5,return_distance=False),y_scop))

from sklearn.ensemble import RandomForestClassifier
lr = RandomForestClassifier()
lr.fit(X_train,y_train)
print(lr.score(X_train,y_train))
print(lr.score(X_scop,y_scop))
