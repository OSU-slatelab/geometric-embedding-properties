# coding=utf-8
from progressbar import progressbar
from tqdm import tqdm

import multiprocessing as mp
import pandas as pd
import numpy as np

import pyemblib
import scipy
import queue
import time
import sys
import os 

'''
preprocessing.py

Preprocessing methods for cuto.py. 
'''

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

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

#========1=========2=========3=========4=========5=========6=========7==

# pass None to vocab to use use entire embedding
# RETURNS: [numpy matrix of word vectors, df of the labels]
def get_embedding_dict(emb_path, emb_format, first_n, vocab):

    print("Preprocessing. ")
    file_name_length = len(emb_path)
    extension = os.path.basename(emb_path).split('.')[-1]

    # Decide if it's a binary or text embedding file, and read in
    # the embedding as a dict object, where the keys are the tokens
    # (strings), and the values are the components of the corresponding 
    # vectors (floats).
    embedding = {}
    read_mode = None
    if first_n == 0 or emb_format == pyemblib.Format.Glove:
        
        print("No value passed for first_n or feature not supported. ")
        first_n = None
    if extension == 'bin':
        read_mode = pyemblib.Mode.Binary
        binary = True
        print("binary reac.")
    elif extension == 'txt':
        read_mode = pyemblib.Mode.Text
        binary = False
        print("text read.")
    else:
        print("Unsupported embedding mode. ")
        exit()
    ''' 
    if emb_format == pyemblib.Format.Glove:
        embedding = loadGloveModel(emb_path)
    '''
    
    if first_n:    
        embedding = pyemblib.read(  emb_path, 
                                    format=emb_format,
                                    mode=read_mode,
                                    first_n=first_n,
                                    replace_errors=True,
                                    skip_parsing_errors=True,
                                    ) 
    else:
        embedding = pyemblib.read(  emb_path, 
                                    format=emb_format,
                                    mode=read_mode,
                                    replace_errors=True,
                                    skip_parsing_errors=True,
                                    ) 
       
    return embedding

#========1=========2=========3=========4=========5=========6=========7==

# pass None to vocab to use use entire embedding
# RETURNS: [numpy matrix of word vectors, df of the labels]
def process_embedding(emb_path, emb_format, first_n, vocab):

    print("Preprocessing. ")
    file_name_length = len(emb_path)
    extension = os.path.basename(emb_path).split('.')[-1]

    # Decide if it's a binary or text embedding file, and read in
    # the embedding as a dict object, where the keys are the tokens
    # (strings), and the values are the components of the corresponding 
    # vectors (floats).
    embedding = {}
    read_mode = None
    if first_n == 0 or emb_format == pyemblib.Format.Glove:
        
        print("No value passed for first_n or feature not supported. ")
        first_n = None
    if extension == 'bin':
        read_mode = pyemblib.Mode.Binary
        binary = True
        print("binary reac.")
    elif extension == 'txt':
        read_mode = pyemblib.Mode.Text
        binary = False
        print("text read.")
    else:
        print("Unsupported embedding mode. ")
        exit()
    ''' 
    if emb_format == pyemblib.Format.Glove:
        embedding = loadGloveModel(emb_path)
    '''
    
    if first_n:    
        embedding = pyemblib.read(  emb_path, 
                                    format=emb_format,
                                    mode=read_mode,
                                    first_n=first_n,
                                    replace_errors=True,
                                    skip_parsing_errors=True,
                                    ) 
    else:
        embedding = pyemblib.read(  emb_path, 
                                    format=emb_format,
                                    mode=read_mode,
                                    replace_errors=True,
                                    skip_parsing_errors=True,
                                    ) 
       
     
    
    # take a subset of the vocab
    new_embedding = {}
    if (vocab != None):
        for word in vocab:
            if word in embedding:
                vector = embedding[word]
                new_embedding.update({word:vector})
        embedding = new_embedding

    # convert embedding to pandas dataframe
    # "words_with_friends" is the column label for the vectors
    # this df has shape [num_inputs,2] since the vectors are all in 1
    # column as length d lists 
    emb_array = np.array(embedding.items())
    print("numpy")
    sys.stdout.flush() 

    label_array = np.array([ row[0] for row in emb_array.tolist() ])
    #print(label_array[0:10])
    sys.stdout.flush() 
    
    vectors_matrix = np.array([ row[1:] for row in emb_array.tolist() ])
    vectors_matrix = np.array([ row[0] for row in vectors_matrix ])
    #print(vectors_matrix[0:10])
    sys.stdout.flush() 

    '''
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
    '''

    return vectors_matrix, label_array

#========1=========2=========3=========4=========5=========6=========7==

# pass None to vocab to use use entire embedding
# DOES: Saves the first n words in a new embedding file
def subset_embedding(emb_path, first_n, vocab):

    print("Preprocessing. ")
    file_name_length = len(emb_path)
    last_char = emb_path[file_name_length - 1]

    # Decide if it's a binary or text embedding file, and read in
    # the embedding as a dict object, where the keys are the tokens
    # (strings), and the values are the components of the corresponding 
    # vectors (floats).
    embedding = {}
    if (last_char == 'n'):
        embedding = pyemblib.read(emb_path, 
                                  mode=pyemblib.Mode.Binary,
                                  first_n=first_n) 
    elif (last_char == 't'):
        embedding = pyemblib.read(emb_path, 
                                  mode=pyemblib.Mode.Text, 
                                  first_n=first_n)
    else:
        print("Unsupported embedding format. ")
        exit()

    # make sure it has a valid file extension
    extension = emb_path[file_name_length - 4:file_name_length]
    if extension != ".txt" and extension != ".bin":
        print("Invalid file path. ")
        exit()
   
    # get the emb_path without the file extension 
    path_no_ext = emb_path[0:file_name_length - 4]
    new_path = path_no_ext + "_SUBSET.txt"

    # write to text embedding file
    pyemblib.write(embedding, 
                   new_path, 
                   mode=pyemblib.Mode.Text)
    
    return 
