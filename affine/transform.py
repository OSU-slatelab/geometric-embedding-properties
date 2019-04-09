from preprocessing  import process_embedding
from preprocessing  import check_valid_file
from preprocessing  import check_valid_dir
from config import get_config

import multiprocessing              as mp
import pandas                       as pd
import numpy                        as np

from progressbar    import progressbar
from tqdm           import tqdm

import datetime
import pyemblib
import scipy
import queue
import time
import sys
import os

'''
transform.py

Script to generate a sequence of affine transformations of a set of 
pretrained word embeddings. 
'''

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: a tuple of the script arguments
def parse_args():

    emb_path = sys.argv[1]
    emb_format = sys.argv[2]  # 'Word2Vec' or 'Glove' 
 
    if len(sys.argv) > 3:
        first_n = sys.argv[3]
    else:
        first_n = 0

    args = [emb_path, emb_format, first_n]
    return args

#========1=========2=========3=========4=========5=========6=========7==

def genflow(emb_path, emb_format, first_n):

    print_sleep_interval = 1 
    print("checkpoint 1")
    check_valid_file(emb_path)
    sys.stdout.flush()

    source_name = os.path.splitext(os.path.basename(emb_path))[0]
    print("Source name:", source_name)
    sys.stdout.flush()

    # take the first n most frequent word vectors for a subset
    # set to 0 to take entire embedding
    first_n = 0

    # Preprocess.
    print("About to preprocess. ") 
    sys.stdout.flush()
    vectors_matrix,label_df = process_embedding(emb_path,
                                                emb_format, 
                                                first_n,
                                                None)
    print("Done preprocessing. ")
    sys.stdout.flush()
    # We get the dimensions of the input dataset. 
    shape = vectors_matrix.shape
    print("Shape of embedding matrix: ", shape)
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()

    # number of rows in the embedding 
    num_inputs = shape[0]
    num_outputs = num_inputs 

    # dimensionality of the embedding file
    dim = shape[1]
 
    #===================================================================

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H%M")
    
    # The name of the embedding to save. 
    parent = os.path.abspath(os.path.join(emb_path, "../"))
    check_valid_dir(parent)

    print("Is anything happening here?") 
    sys.stdout.flush()
    transforms = get_config(dim)
    print("Got transforms. ")
    sys.stdout.flush()

    output_embedding_paths = [] 

    for i,transform in tqdm(enumerate(transforms)):
        
        func = transform[0]
        arglist = transform[1]

        new_emb_path =  str(os.path.join(parent, "affine-" + str(i) + "__source--" + source_name 
                        + "__" + "time--" + timestamp + ".bin"))
        sys.stdout.flush()
        output_embedding_paths.append(new_emb_path)
    
        print("About to start generation.")
        sys.stdout.flush()
        transformed_vectors = func(vectors_matrix, arglist) 
        
        # shape [<num_inputs>,<dimensions>]
        print("labels shape: ", label_df.shape)
        sys.stdout.flush()
        
        # creates the emb dict
        dist_emb_dict = {}
        for i in tqdm(range(len(label_df))):
            emb_array_row = transformed_vectors[i]
            dist_emb_dict.update({label_df[i]:emb_array_row})
            sys.stdout.flush()

        print("Embedding dict created. ")
        sys.stdout.flush()
        
        # saves the embedding
        pyemblib.write(dist_emb_dict, 
                       new_emb_path, 
                       mode=pyemblib.Mode.Binary)

        print("Embedding saved to: " + new_emb_path)

    # Write the output embedding names to a text file. 
    outputlist_name = "affine-outputlist__source--" + source_name + "__time--" + timestamp + ".txt"
    outputlist_path = os.path.join(parent, outputlist_name)
    with open(outputlist_path, 'w') as f:
        for path in output_embedding_paths:
            f.write(path + "\n")

    return

#========1=========2=========3=========4=========5=========6=========7==

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here 
    
    args = parse_args()

    emb_path = args[0]
    emb_format = args[1]
    first_n = args[2]   
 
    genflow(emb_path, emb_format, first_n) 


