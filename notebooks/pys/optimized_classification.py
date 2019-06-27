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

import pickle

# Get and parse all pdb files in a folder
def parsePdbFiles(dir_path):
    structures = []
    files = os.listdir(dir_path)
    pdb_files = [(f, os.path.join(dir_path, f)) for f in files if f.endswith(".pdb")]

    for pdb, pdb_path in pdb_files:
        parser = PDB.PDBParser()
        structure = parser.get_structure(pdb, pdb_path)
        structures.append(structure)
    return structures

def CreateContactMap(structure,resize_to):
    coords_list = []
    for residue in structure.get_residues():
        try:
            coords = residue['CA'].coord
            coords_list.append(coords)
        except:
            continue
    contact_map = []
    for c in coords_list:
        coord_dist = []
        for c_ in coords_list:
            dist = np.linalg.norm(c-c_) # calculate distance between coordinates of CAs of residues
            coord_dist.append(dist)
        contact_map.append(coord_dist)
    # Resize protein_matrix
    try:
        resized = cv2.resize(np.array(contact_map), (resize_to[0], resize_to[1]), interpolation=cv2.INTER_AREA)
    except:
        resized = None
    return resized

# Removes symmetry from the contact map
def RemoveSymmetry(matrix):
    flatten = []
    row = 0
    for i in range(1, len(matrix)):
        flatten += matrix[row][i:].tolist()
        row+=1
    return np.array(flatten)

# get structure list and returns protein, contact map dictionary
def ProteinContactMapDict(structures, resize_to=(32,32), removeSymmetry=True):
    protein_matrix_dict = {}
    for protein in structures:
        protein_matrix = CreateContactMap(protein, resize_to)
        if type(protein_matrix) == np.ndarray:
            if removeSymmetry == True:
                protein_matrix = RemoveSymmetry(protein_matrix)
            else:
                protein_matrix = protein_matrix.flatten()
            protein_matrix_dict[protein.id] = protein_matrix
    return protein_matrix_dict

def pipeline(dir_path, batch_size, resizeto=64, n_hidden=500, n_iteration=100, learning_rate = 0.0001):

    pdbs = os.listdir(dir_path)
    n_pdbs = len(pdbs)
    print("Total number of pdbs: %d" %n_pdbs)

    # Setting Linear Autoencoder parameters
    n_inputs = (resizeto**2/2)-(resizeto/2) # input is flatten version of input matrix
    n_hidden = n_hidden
    n_outputs = n_inputs

    learning_rate = learning_rate

    X = tf.placeholder(tf.float32, shape=[None, n_inputs])
    hidden = fully_connected(X, n_hidden, activation_fn=None)
    outputs = fully_connected(hidden, n_outputs, activation_fn=None)

    reconstruction_loss = tf.reduce_sum(tf.square(outputs - X)) # MSE

    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(reconstruction_loss)

    init = tf.global_variables_initializer()

    n_iterations = n_iteration # Number of iterations
    codings = hidden # the output of the hidden layer provides the codings

    print("images resized to: %dx%d" %(resizeto,resizeto))
    print("input size: %d" %n_inputs)
    print("hidden input size: %d" %n_hidden)
    print("learning rate: %d" %learning_rate)
    print("number of iterations: %d" %n_iteration)

    # Train
    with tf.Session() as sess:
        init.run()
        loss = []
        for iteration in range(n_iterations):
            print("\n--- Iteration %d ---" %iteration)
            start=0
            end = batch_size
            iteration_loss = []
            for i in range(n_pdbs/batch_size+1):
                batch_pdbs = pdbs[start:end]
                print("%d - Parsing %d pdbs" %(i, len(batch_pdbs)))
                start+=batch_size
                end+=batch_size

                # Constracting Pdb Structures
                #print("Constracting pdb structures, Generating Protein Contact Maps and Training")
                structures = []
                pdb_files = [(f, os.path.join(dir_path, f)) for f in batch_pdbs if f.endswith(".pdb")]

                for pdb, pdb_path in pdb_files:
                    parser = PDB.PDBParser()
                    structure = parser.get_structure(pdb, pdb_path)
                    structures.append(structure)

                # Generating Protein Contact Maps
                proteinmatrixdict = ProteinContactMapDict(structures, resize_to=(resizeto,resizeto), removeSymmetry=True)

                labels, features = proteinmatrixdict.keys(), proteinmatrixdict.values()

                _, loss_val = sess.run([training_op, reconstruction_loss], feed_dict={X: features}) # no labels (unsupervised)
                iteration_loss.append(loss_val)
            loss.append(sum(iteration_loss)/float(n_pdbs))
        print("---------- Training end ----------\n\n")
        # LOSS GRAPH
        fig, ax = plt.subplots()
        ax.plot(range(0,len(loss)), loss, 'go-', linewidth=1, markersize=1)
        fig.savefig("loss.png")

        # Test
        print("---------- Test starts ----------")
        new_features = []
        start=0
        end = batch_size

        for i in range(n_pdbs/batch_size+1):
            batch_pdbs = pdbs[start:end]
            print("%d - Parsing %d pdbs" %(i, len(batch_pdbs)))
            start+=batch_size
            end+=batch_size

            # Constracting Pdb Structures
            structures = []
            pdb_files = [(f, os.path.join(dir_path, f)) for f in batch_pdbs if f.endswith(".pdb")]

            for pdb, pdb_path in pdb_files:
                parser = PDB.PDBParser()
                structure = parser.get_structure(pdb, pdb_path)
                structures.append(structure)

            # Generating Protein Contact Maps
            proteinmatrixdict = ProteinContactMapDict(structures, resize_to=(resizeto,resizeto), removeSymmetry=True)

            labels, features = proteinmatrixdict.keys(), proteinmatrixdict.values()

            codings_val = codings.eval(feed_dict={X: features})
            new_features += (zip(labels, codings_val))

        with open("newnewfeatures.py", "wb") as f:
            pickle.dump(dict(new_features), f)

pipeline("PDBs",batch_size=100,n_iteration=100)
