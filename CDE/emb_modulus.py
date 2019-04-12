import tensorflow.contrib.layers as lays
import multiprocessing as mp
import tensorflow as tf 
import pandas as pd
import numpy as np

from tqdm import tqdm

import pyemblib
import scipy
import time
import sys
import os

'''
emb_modulus.py 

Compute and print the average vector norm given the location of a
pretrained embedding.  
'''


#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: a tuple of the script arguments
def parse_args():

    emb_path = sys.argv[1]

    args = [emb_path,
           ]

    return args

#========1=========2=========3=========4=========5=========6=========7==

def check_valid_dir(some_dir):
    if not os.path.isdir(some_dir):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")
        print("DIES IST EIN UNGÜLTIGES VERZEICHNIS!!!!")
        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()

#========1=========2=========3=========4=========5=========6=========7==

def check_valid_file(some_file):
    if not os.path.isfile(some_file):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("")
        print("DIES IST KEIN GÜLTIGER SPEICHERORT FÜR DATEIEN!!!!")
        print("")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        exit()

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS a tuple of the vectors and the labels dataframe
def process_embedding(emb_path):

    print("Preprocessing. ")
    file_name_length = len(emb_path)
    last_char = emb_path[file_name_length - 1]

    # Decide if it's a binary or text embedding file, and read in
    # the embedding as a dict object, where the keys are the tokens
    # (strings), and the values are the components of the corresponding 
    # vectors (floats).
    embedding = {}
    if (last_char == 'n'):
        embedding = pyemblib.read(emb_path, mode=pyemblib.Mode.Binary)
    elif (last_char == 't'):
        embedding = pyemblib.read(emb_path, mode=pyemblib.Mode.Text)
    else:
        print("Unsupported embedding format. ")
        exit()

    # convert embedding to pandas dataframe
    # "words_with_friends" is the column label for the vectors
    # this df has shape [num_inputs,2] since the vectors are all in 1
    # column as length d lists 
    emb_df = pd.Series(embedding, name="words_with_friends")
    # print(emb_df.head(10))

    # reset the index of the dataframe
    emb_df = emb_df.reset_index()
    # print(emb_df.head(10))

    # matrix of just the vectors
    emb_matrix = emb_df.words_with_friends.values.tolist()
    # print(emb_matrix[0:10])

    # dataframe of just the vectors
    vectors_df = pd.DataFrame(emb_matrix,index=emb_df.index)
    # print(vectors_df.head(10))

    # numpy matrix of just the vectors
    vectors_matrix = vectors_df.as_matrix()
    # print(vectors_matrix[0:10])

    return vectors_matrix, emb_df.loc[:,"index"]

#========1=========2=========3=========4=========5=========6=========7== 

def compute_modulus(vectors_matrix):

    norm_array = np.linalg.norm(vectors_matrix,axis=1)
    print("norm_array:", norm_array)
    average_norm = np.average(norm_array)

    return average_norm

#========1=========2=========3=========4=========5=========6=========7== 

def runflow(emb_path):

    unit_norm = True
    # unit_norm = False

    # PREPROCESSING
    check_valid_file(emb_path) 
    vectors_matrix,label_df = process_embedding(emb_path)

    # We get the dimensions of the input dataset. 
    shape = vectors_matrix.shape
    print("Shape of embedding matrix: ", shape)

    # number of rows in the embedding 
    num_inputs = shape[0]
    num_outputs = num_inputs 

    # dimensionality of the embedding file
    num_hidden = shape[1]

    # unit norm option
    if (unit_norm):

        print("Unit norming the embedding. ")
        norms_matrix = np.linalg.norm(vectors_matrix, axis=1)
        # norms_matrix[norms_matrix==0] = 1
        vectors_matrix = vectors_matrix / np.expand_dims(norms_matrix, -1)

    modulus = compute_modulus(vectors_matrix)
    print("The average vector norm is: ", modulus)

    return

#========1=========2=========3=========4=========5=========6=========7== 

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here 
    
    args = parse_args()

    emb_path = args[0]
    
    runflow(emb_path) 
