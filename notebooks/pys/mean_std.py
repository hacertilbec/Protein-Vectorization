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



def scale_features(features,mean,std_dev):
    return (np.array(features)-mean)/std_dev


for fold in range(1,2):
    train_pdbs = []
    with open("Folds/{n}_train.txt".format(n=fold),"r") as f:
        for i in f.readlines():
            train_pdbs.append(i.strip("\n"))
    test_pdbs = []
    with open("Folds/{n}_test.txt".format(n=fold),"r") as f:
        for i in f.readlines():
            test_pdbs.append(i.strip("\n"))

    print(len(train_pdbs))
    print(len(test_pdbs))

    train_features, test_features = [],[]
    train_pdb_samples, test_pdb_samples = [],[]


    for i in range(int(len(train_pdbs)/batch_size)+1): # BATCH
        structures = []
        for pdb in train_pdbs[i*batch_size: (i+1)*batch_size]: # EACH PDB IN THE BATCH
            if pdb == ".ipynb_checkpoints":
                continue
            pdb_path = os.path.join("PDBs", pdb)
            parser = PDB.PDBParser()
            structure = parser.get_structure(pdb, pdb_path)
            structures.append(structure)
        matrixdict = DistanceMatrixDict(structures, resize_strategy=strategy, resize_to=(filter_size,filter_size),removeSymmetry=True)
        pdbs, features = list(matrixdict.keys()), list(matrixdict.values())

        train_pdb_samples+=pdbs
        train_features+=features

    input_size = len(train_features[0])

    # SCALING
    train_features = np.array(train_features)
    mean = np.mean(train_features.flatten())
    std_dev = np.std(train_features.flatten())
    print(mean)
    print(std_dev)
