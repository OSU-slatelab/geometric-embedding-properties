from preprocessing  import process_embedding
from preprocessing  import subset_embedding
from preprocessing  import check_valid_file
from preprocessing  import check_valid_dir

import multiprocessing              as mp
import tensorflow                   as tf 
import pandas                       as pd
import numpy                        as np

from progressbar    import progressbar
from tqdm           import tqdm

import pyemblib
import scipy
import queue
import time
import sys
import os 

'''
save_first_n.py

Script to save the first n most frequent words in an embedding file. 
'''

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: a tuple of the script arguments
def parse_args():

    emb_path = sys.argv[1]
    first_n = int(sys.argv[2])    

    args = [emb_path,
            first_n,
            ]

    return args

#========1=========2=========3=========4=========5=========6=========7==

def saveflow(emb_path,first_n):

    check_valid_file(emb_path)
   
    subset_embedding(emb_path,first_n,None)

#========1=========2=========3=========4=========5=========6=========7==

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here 
    
    args = parse_args()

    emb_path = args[0]
    first_n = args[1]
    
    saveflow(emb_path,first_n) 


