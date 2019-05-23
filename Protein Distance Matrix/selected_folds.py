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

def scale_features(features,mean,std_dev):
    return (np.array(features)-mean)/std_dev

def calculate_mean_std(data):
    features = np.array(data)
    mean = np.mean(features.flatten())
    std_dev = np.std(features.flatten())
    return mean, std_dev

def create_fdict(pdbs,features):
    feature_dict = {}
    for i in enumerate(pdbs):
        if "sample" in i[1]:
            pdb = i[1].split("sample")[0]
        else:
            pdb = i[1]
        feature_dict.setdefault(pdb,[])
        feature_dict[pdb].append(features[i[0]])
    return feature_dict

def X_y(feature_dict,label_dict):
    X_train,y_train = [], []
    for pdb,vector in feature_dict.items():
        pdb = pdb.split(".")[0]
        X_train.append(np.average(vector,axis=0))
        y_train.append(".".join(label_dict[pdb].split(".")[:2]))
    return np.array(X_train),y_train


class Autoencoder:
    def __init__(self):
        self.sess = None

    def build_model(self):
        # AUTOENCODER
        tf.reset_default_graph()
        tf.set_random_seed(42)

        self.X = tf.placeholder(tf.float32, shape=[None, input_size])
        self.hidden = fully_connected(self.X, encoding_size, activation_fn=tf.nn.elu)
        outputs = fully_connected(self.hidden, input_size, activation_fn=None)

        reconstruction_loss = tf.reduce_sum(tf.square(outputs - self.X)) # MSE

        optimizer = tf.train.AdamOptimizer(0.0001)
        self.training_op = optimizer.minimize(reconstruction_loss)

        init = tf.global_variables_initializer()

        self.sess = tf.InteractiveSession()
        self.sess.run(init)

    def train(self,data,num_iters=100,batch_size=100):
        for iteration in range(num_iters):
            for i in range(int(len(data)/batch_size)+1): # BATCH
                batch = data[i*batch_size: (i+1)*batch_size]
                self.sess.run([self.training_op], feed_dict={self.X: batch})

    def encode(self,sample):
        return self.sess.run([self.hidden], feed_dict={self.X: sample})[0]

def loadSCOPTest():
    # SCOP TEST PDBs
    scop_pdbs = []
    with open("Folds/6folds_scop.txt","r") as f:
        for i in f.readlines():
            scop_pdbs.append(i.strip("\n"))

    structures = []
    for pdb in scop_pdbs:
        pdb=pdb+".pdb"
        if pdb == ".ipynb_checkpoints":
            continue
        pdb_path = os.path.join("SCOP_Test", pdb)
        parser = PDB.PDBParser()
        structure = parser.get_structure(pdb, pdb_path)
        structures.append(structure)

    matrixdict = DistanceMatrixDict(structures, resize_strategy=strategy, resize_to=(filter_size,filter_size),removeSymmetry=True)
    scop_pdb_samples, scop_features = list(matrixdict.keys()), list(matrixdict.values())
    return scop_pdb_samples, scop_features


# TRAIN AND TEST
train_acc_1,test_acc_1, scop_acc_1 = [],[],[]
train_acc_5,test_acc_5, scop_acc_5 = [],[],[]

train_pdbs = []
with open("Folds/6folds_train.txt","r") as f:
    for i in f.readlines():
        train_pdbs.append(i.strip("\n"))
test_pdbs = []
with open("Folds/6folds_test.txt","r") as f:
    for i in f.readlines():
        test_pdbs.append(i.strip("\n"))

print(len(train_pdbs))
print(len(test_pdbs))

train_features_, test_features_ = [],[]
train_pdb_samples, test_pdb_samples = [],[]

# TRAIN FEATURES
for pdb in train_pdbs: # EACH PDB IN THE BATCH
    pdb=pdb+".pdb"
    if pdb == ".ipynb_checkpoints":
        continue
    pdb_path = os.path.join("PDBs", pdb)
    parser = PDB.PDBParser()
    structure = parser.get_structure(pdb, pdb_path)
    matrixdict = DistanceMatrixDict([structure], resize_strategy=strategy, resize_to=(filter_size,filter_size),removeSymmetry=True)
    pdbs, features = list(matrixdict.keys()), list(matrixdict.values())
    train_pdb_samples+=pdbs
    train_features_+=features

input_size = len(train_features_[0])

# SCALING
mean, std_dev = calculate_mean_std(train_features_)
train_features = scale_features(train_features_,mean,std_dev)

# AUTOENCODER
autoencoder = Autoencoder() # create an autoencoder model
autoencoder.build_model() # build model and run session
autoencoder.train(train_features,num_iters=100,batch_size=100) # train the model

# TEST FEATURES
for pdb in test_pdbs: # EACH PDB IN THE BATCH
    pdb=pdb+".pdb"
    if pdb == ".ipynb_checkpoints":
        continue
    pdb_path = os.path.join("PDBs", pdb)
    parser = PDB.PDBParser()
    structure = parser.get_structure(pdb, pdb_path)
    matrixdict = DistanceMatrixDict([structure], resize_strategy=strategy, resize_to=(filter_size,filter_size),removeSymmetry=True)
    pdbs, features = list(matrixdict.keys()), list(matrixdict.values())
    test_pdb_samples+=pdbs
    test_features_+=features

# SCALING
test_features = scale_features(test_features_,mean,std_dev)

# REPORT
print("number of train pdbs: {n}".format(n=len(train_pdb_samples)))
print("number of test pdbs: {n}".format(n=len(test_pdb_samples)))

# Embedding vectors of train and test set
new_train_features = autoencoder.encode(train_features)
new_test_features = autoencoder.encode(test_features)

# Create pdb:feature dictionaries
train_feature_dict = create_fdict(train_pdb_samples,new_train_features)
test_feature_dict = create_fdict(test_pdb_samples,new_test_features)

# X and y sets are prepared
X_train, y_train = X_y(train_feature_dict,label_dict)
X_test, y_test = X_y(test_feature_dict,label_dict)

# Encoding fold labels
uniques = list(set(y_train).union(set(y_test)))
group2id = dict(zip(uniques, range(len(uniques))))
y_train = np.array(list(map(lambda x: group2id[x], y_train)))
y_test = np.array(list(map(lambda x: group2id[x], y_test)))


# CLASSIFICATION
print("KNN K=1")
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
print(knn.score(X_train,y_train))
print(knn.score(X_test,y_test))
print("KNN K=5")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
print(knn.score(X_train,y_train))
print(knn.score(X_test,y_test))


print("----SCOP TEST")
all_features = train_features_ + test_features_
all_pdb_samples = train_pdb_samples + test_pdb_samples

mean, std_dev = calculate_mean_std(all_features)
all_features = scale_features(all_features,mean,std_dev)
print("all pdbs: {n}".format(n=len(all_features)))

# AUTOENCODER
autoencoder = Autoencoder() # create an autoencoder model
autoencoder.build_model() # build model and run session
autoencoder.train(all_features,num_iters=100,batch_size=100) # train the model
all_features = autoencoder.encode(all_features)
all_feature_dict = create_fdict(all_pdb_samples,all_features)
X, y = X_y(all_feature_dict,label_dict)

# LOAD SCOPTest pdbs
scop_pdb_samples, scop_features = loadSCOPTest()
print(len(scop_features))

scop_features = scale_features(scop_features,mean,std_dev)
new_scop_features = autoencoder.encode(scop_features)
scop_feature_dict = create_fdict(scop_pdb_samples,new_scop_features)
X_scop, y_scop = X_y(scop_feature_dict,scop_label_dict)

# Encoding fold labels
uniques = list(set(y).union(set(y_scop)) )
group2id = dict(zip(uniques, range(len(uniques))))


y_all = np.array(list(map(lambda x: group2id[x], y)))
y_scop = np.array(list(map(lambda x: group2id[x], y_scop)))

# CLASSIFICATION
print("KNN K=1")
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,y)
print(knn.score(X,y))
print(knn.score(X_scop, y_scop))

print("KNN K=5")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
print(knn.score(X,y))
print(knn.score(X_scop, y_scop))