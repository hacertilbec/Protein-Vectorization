from pdb_utils import *

import pickle
import numpy as np
import pandas as pd
import random

from functools import partial
import sys

from Bio import PDB
import os
import cv2


import time


# Loading label dictionaries
with open('pickle files/fold_groups.pkl', 'rb') as f:
    fold_dict = pickle.load(f)



# Distance Matrices
all_features = []

for fold,pdb_list in list(fold_dict.items()):
    structures = []
    for pdb in pdb_list:
        try:                
            pdb_path = os.path.join("PDBs", pdb+".pdb")
            parser = PDB.PDBParser()
            structure = parser.get_structure(pdb, pdb_path)
            structures.append(structure)
        except:
            continue
    matrix = DistanceMatrixDict(structures, resize_strategy="strategy1", resize_to=(64,64),removeSymmetry=True)
    pdb_names, features = list(matrix.keys()), list(matrix.values())
    all_features += features
    
all_features=np.array(all_features)
print(all_features.shape)
from sklearn.decomposition import IncrementalPCA
transformer = IncrementalPCA(n_components=95, batch_size=100)
transformer.fit(all_features)

print(transformer.transform(all_features).shape)

from joblib import dump
dump(transformer, 'pca_with_PDBs.joblib') 

