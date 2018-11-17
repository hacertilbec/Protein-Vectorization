from Bio import PDB
import numpy as np
import os
import cv2
from math import ceil
import random

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

# Sampling: For a protein with length over 256, they randomly sampled a 256x256
# sub-matrix from its distance matrix. They repeated this procedure
# multiple times and obtained an ensemble

def sampling(distance_matrix, sample_size, new_shape=(64,64)):
    if not sample_size:
        sample_size = int(ceil((distance_matrix.shape[0]-new_shape[0])**2/10.)) # Here, the number of ensemble matrices 
    ensemble = []                                              # was set to be proportional to the length of query protein
    for sample in range(sample_size):
        sampled_matrix = []
        x,y = random.randint(0,len(distance_matrix)-new_shape[0]), random.randint(0,len(distance_matrix)-new_shape[0])
        for i in range(x,x+new_shape[0]):
            sampled_matrix.append(distance_matrix[i][y:y+new_shape[0]])
        ensemble.append(sampled_matrix)
    return(np.array(ensemble))

# Padding: For a protein with length smaller than 256, we embedded its distance
# matrix into a 256x256 matrix with all elements being 0. The embedding
# positions are random; thus, we obtained an ensemble of 256x256 matrices
# after repeating this operation multiple times.

def padding(distance_matrix, sample_size, new_shape=(64,64)):
    if not sample_size:
        sample_size = int(ceil((distance_matrix.shape[0]-new_shape[0])**2/10.)) # Here, the number of ensemble matrices 
    ensemble = []                                                       # was set to be proportional to the
    for sample in range(sample_size):                                   # length of query protein
        sampled_matrix = [[0 for i in range(new_shape[0])] for i in range(new_shape[0])]
        x,y = random.randint(0,len(sampled_matrix)-len(distance_matrix)), random.randint(0,len(sampled_matrix)-len(distance_matrix))
        s = 0
        for i in range(x,x+len(distance_matrix)):
            sampled_matrix[i][y:y+len(distance_matrix)] = distance_matrix[s][:]
            s+=1
        ensemble.append(sampled_matrix)
    return(np.array(ensemble))

def createDistanceMatrix(structure,resize_strategy,resize_to,sample_size):
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
    distance_matrix = np.array(distance_matrix)
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
        else:
            print("Not a valid strategy method. Use False, strategy1, or strategy2.")
            return
    return resized

# Removes symmetry from the distance matrix
def RemoveSymmetry(matrix):
    flatten = []
    row = 0
    for i in range(1, len(matrix)):
        flatten += matrix[row][i:].tolist()
        row+=1
    return np.array(flatten)

# get structure list and returns protein, distance matrix dictionary
def DistanceMatrixDict(structures,resize_strategy="strategy1", resize_to=(64,64), removeSymmetry=False,sample_size=None):
    if resize_strategy == "strategy2" and removeSymmetry == True:
        print("RemoveSymmetry parameter can not be used with strategy2")
        return
    protein_matrix_dict = {}
    for protein in structures:
        protein_matrix = createDistanceMatrix(protein,resize_strategy, resize_to=resize_to,sample_size=sample_size)
        if resize_strategy == "strategy2":
            if protein_matrix.shape[0] == resize_to[0]:
                protein_matrix_dict[protein.id] = protein_matrix.flatten()
            else:
                for sample in enumerate(protein_matrix):
                    key = protein.id + "sample" + str(sample[0])
                    protein_matrix_dict[key] = sample[1].flatten()
        else:
            if type(protein_matrix) == np.ndarray:
                if removeSymmetry == True:
                    protein_matrix = RemoveSymmetry(protein_matrix)
                else:
                    protein_matrix = protein_matrix.flatten()
                protein_matrix_dict[protein.id] = protein_matrix
    return protein_matrix_dict
