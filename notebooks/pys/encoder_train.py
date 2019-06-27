#from autoencoders import *
from pdb_utils import *
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

from keras.layers import Input, Dense
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras import regularizers
from keras import backend as K
from keras.optimizers import Adam, Adadelta
K.tensorflow_backend._get_available_gpus()

import matplotlib

import matplotlib.pyplot as plt

filter_size=64
strategy = "strategy1"
batch_size = 100
num_iters = 1000
input_size=2016


def build_model(inp_dim,encoding_dim,sparse=False):
    # this is our input placeholder
    input_img = Input(shape=(inp_dim,))
    # "encoded" is the encoded representation of the input
    if sparse==True:
        encoded = Dense(encoding_dim, activation='relu',activity_regularizer=regularizers.l1(10e-5))(input_img)
    else:
        encoded = Dense(encoding_dim, activation='relu')(input_img)
    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(inp_dim, activation='sigmoid')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input_img, decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input_img, encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(encoded_input, decoder_layer(encoded_input))
    optimizer = Adadelta(lr=0.001)
    autoencoder.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy','mse'])
    autoencoder.summary()
    return encoder,decoder,autoencoder


fold=1
train_pdbs = []
with open("Folds/{n}_train.txt".format(n=fold),"r") as f:
    for i in f.readlines():
        train_pdbs.append(i.strip("\n"))
with open("Folds/{n}_test.txt".format(n=fold),"r") as f:
    for i in f.readlines():
        train_pdbs.append(i.strip("\n"))
        
train_pdbs = train_pdbs
print(len(train_pdbs))

train_features = []
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
    train_features+=features

# TRAIN MODEL
X_train = np.array(train_features)
inp_dim = X_train.shape[1]
enc_dim = 100
encoder,decoder,autoencoder = build_model(inp_dim,enc_dim)
# Set callback functions to early stop training and save the best model so far
callbacks = [EarlyStopping(monitor='val_loss', patience=2),
         ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

hist = autoencoder.fit(X_train, X_train,
            epochs=num_iters,
            batch_size=batch_size,
            callbacks=callbacks, # Early stopping
            shuffle=True,
            validation_split=0.3
            )
encoder.save("encoder.h5")
# LOSS GRAPH
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']

fig, ax = plt.subplots()
ax.plot(range(0,len(train_loss)), train_loss, 'go-', linewidth=1, markersize=1,color="red")
ax.plot(range(0,len(val_loss)), val_loss, 'go-', linewidth=1, markersize=1,color="blue")
plt.savefig("LOSS.png")
