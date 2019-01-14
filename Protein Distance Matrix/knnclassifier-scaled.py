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
    distance_matrix = (np.array(distance_matrix)-mean)/std_dev
    # Resize protein_matrix
    if resize_strategy == False or len(distance_matrix) == resize_to[0]:
        return distance_matrix
    else:
        if resize_strategy == "strategy1":
            try:
                resized = cv2.resize(distance_matrix, (resize_to[0], resize_to[1]), interpolation=cv2.INTER_AREA)
            except:
                return []
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
        if len(protein_matrix) == 0:
            continue
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


with open('pickle files/label_dict.pkl', 'rb') as f:
    label_dict = pickle.load(f)
with open('pickle files/test_labels.pkl', 'rb') as f:
    scop_label_dict = pickle.load(f)

# SCOP TEST PDBs
structures = []
for pdb in os.listdir("SCOP_Test/"):
    if not pdb.endswith(".pdb"):
        continue
    pdb_path = os.path.join("SCOP_Test", pdb)
    parser = PDB.PDBParser()
    structure = parser.get_structure(pdb, pdb_path)
    structures.append(structure)
matrixdict = DistanceMatrixDict(structures, resize_strategy=strategy, resize_to=(filter_size,filter_size),removeSymmetry=True)
scop_pdb_samples, scop_features = list(matrixdict.keys()), list(matrixdict.values())


train_acc_1,test_acc_1, scop_acc_1 = [],[],[]
train_acc_5,test_acc_5, scop_acc_5 = [],[],[]

for fold in range(1,11):
    train_pdbs = []
    with open("Folds/{n}_train.txt".format(n=fold),"r") as f:
        for i in f.readlines():
            train_pdbs.append(i.strip("\n"))
    test_pdbs = []
    with open("Folds/{n}_test.txt".format(n=fold),"r") as f:
        for i in f.readlines():
            test_pdbs.append(i.strip("\n"))
    train_pdbs = train_pdbs
    test_pdbs = test_pdbs

    print(len(train_pdbs))
    print(len(test_pdbs))

    all_values = []
    for pdb in train_pdbs:
        try:
            pdb_path = os.path.join("PDBs", pdb)
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
    mean = np.mean(all_values)
    std_dev = np.std(all_values)

    print(mean)
    print(std_dev)


    tf.reset_default_graph()
    tf.set_random_seed(42)

    X = tf.placeholder(tf.float32, shape=[None, input_size])
    hidden = fully_connected(X, encoding_size, activation_fn=tf.nn.relu)
    outputs = fully_connected(hidden, input_size, activation_fn=tf.nn.softmax)

    reconstruction_loss = tf.reduce_sum(tf.square(outputs - X)) # MSE

    optimizer = tf.train.AdamOptimizer(0.0001)
    training_op = optimizer.minimize(reconstruction_loss)

    codings = hidden # the output of the hidden layer provides the codings

    init = tf.global_variables_initializer()

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(init)


    model_path = "models/"+strategy+str(filter_size)+str(encoding_size)+".ckpt"

    train_features, test_features = [],[]
    train_pdb_samples, test_pdb_samples = [],[]

    autoencoder = None
    for i in range(int(len(train_pdbs)/batch_size)+1): # BATCH
        structures = []
        for pdb in train_pdbs[i*batch_size: (i+1)*batch_size]: # EACH PDB IN THE BATCH
            if pdb == ".ipynb_checkpoints":
                continue
            pdb_path = os.path.join("PDBs", pdb)
            parser = PDB.PDBParser()
            structure = parser.get_structure(pdb, pdb_path)
            structures.append(structure)
        matrixdict = DistanceMatrixDict2(structures, mean, std_dev, resize_strategy=strategy, resize_to=(filter_size,filter_size),removeSymmetry=True)
        pdbs, features = list(matrixdict.keys()), list(matrixdict.values())

        train_pdb_samples+=pdbs
        train_features+=features
        input_size = len(features[0])

    # ITERATION
    for iteration in range(num_iters):
        for i in range(int(len(train_features)/batch_size)+1): # BATCH
            batch = train_features[i*batch_size: (i+1)*batch_size]
            sess.run([training_op], feed_dict={X: batch})

    for i in range(int(len(test_pdbs)/batch_size)+1): # BATCH
        structures = []
        for pdb in test_pdbs[i*batch_size: (i+1)*batch_size]: # EACH PDB IN THE BATCH
            if pdb == ".ipynb_checkpoints":
                continue
            pdb_path = os.path.join("PDBs", pdb)
            parser = PDB.PDBParser()
            structure = parser.get_structure(pdb, pdb_path)
            structures.append(structure)
        matrixdict = DistanceMatrixDict2(structures, mean, std_dev, resize_strategy=strategy, resize_to=(filter_size,filter_size),removeSymmetry=True)
        pdbs, features = list(matrixdict.keys()), list(matrixdict.values())
        test_pdb_samples+=pdbs
        test_features+=features

    print("number of train pdbs: {n}".format(n=len(train_pdb_samples)))
    print("number of test pdbs: {n}".format(n=len(test_pdb_samples)))
    # Embedding vectors of train and test set
    new_train_features = sess.run([hidden], feed_dict={X: train_features})[0]
    new_test_features = sess.run([hidden], feed_dict={X: test_features})[0]
    new_scop_features = sess.run([hidden], feed_dict={X: scop_features})[0]

    train_feature_dict = {}
    for i in enumerate(train_pdb_samples):
        if "sample" in i[1]:
            pdb = i[1].split("sample")[0]
        else:
            pdb = i[1]
        train_feature_dict.setdefault(pdb,[])
        train_feature_dict[pdb].append(new_train_features[i[0]])

    test_feature_dict = {}
    for i in enumerate(test_pdb_samples):
        if "sample" in i[1]:
            pdb = i[1].split("sample")[0]
        else:
            pdb = i[1]
        test_feature_dict.setdefault(pdb,[])
        test_feature_dict[pdb].append(new_test_features[i[0]])

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

    X_test = []
    y_test = []

    for pdb,vector in test_feature_dict.items():
        pdb = pdb.split(".")[0]
        X_test.append(np.average(vector,axis=0))
        y_test.append(".".join(label_dict[pdb.split(".")[0]].split(".")[:2]))

    X_scop = []
    y_scop = []

    for pdb,vector in scop_feature_dict.items():
        pdb = pdb.split(".")[0]
        X_scop.append(np.average(vector,axis=0))
        y_scop.append(".".join(scop_label_dict[pdb.split(".")[0]].split(".")[:2]))


    uniques = list(set(y_train).union(set(y_test)).union(set(y_scop)) )
    group2id = dict(zip(uniques, range(len(uniques))))

    X_train = np.array(X_train)
    y_train = np.array(list(map(lambda x: group2id[x], y_train)))
    X_test = np.array(X_test)
    y_test = np.array(list(map(lambda x: group2id[x], y_test)))
    X_scop = np.array(X_scop)
    y_scop = np.array(list(map(lambda x: group2id[x], y_scop)))


    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train,y_train)
    train_acc_1.append(knn.score(X_train,y_train))
    test_acc_1.append(knn.score(X_test,y_test))
    scop_acc_1.append(knn.score(X_scop,y_scop))

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train,y_train)
    train_acc_5.append(knn.score(X_train,y_train))
    test_acc_5.append(knn.score(X_test,y_test))
    scop_acc_5.append(knn.score(X_scop,y_scop))


with open("result_{s}_{f}_{e}_scaled.txt".format(s=strategy,f=filter_size,e=encoding_size),"w") as f:
    f.write("strategy:{s}\nfilter size:{f}\nencoding size:{e}\n\n".format(s=strategy,f=filter_size,e=encoding_size))
    f.write("10-fold train accuracy with knn(k=1):{tr}\n".format(tr=sum(train_acc_1)/len(train_acc_1) ))
    f.write("10-fold test accuracy with knn(k=1):{tr}\n".format(tr=sum(test_acc_1)/len(test_acc_1) ))
    f.write("10-fold SCOP accuracy with knn(k=1):{tr}\n\n".format(tr=sum(scop_acc_1)/len(scop_acc_1) ))
    f.write("10-fold train accuracy with knn(k=5):{tr}\n".format(tr=sum(train_acc_5)/len(train_acc_5) ))
    f.write("10-fold test accuracy with knn(k=5):{tr}\n".format(tr=sum(test_acc_5)/len(test_acc_5) ))
    f.write("10-fold SCOP accuracy with knn(k=5):{tr}\n".format(tr=sum(scop_acc_5)/len(scop_acc_5) ))
