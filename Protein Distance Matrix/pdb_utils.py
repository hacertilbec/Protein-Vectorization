from Bio import PDB
import numpy as np
import os
import cv2

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
    model=structure[0]
    for chain in model.get_list():
        for residue in chain.get_list():
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
    if resize_to == False:
        return np.array(contact_map)
    else:
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
