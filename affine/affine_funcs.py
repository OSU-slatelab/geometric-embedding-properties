from preprocessing  import process_embedding
from preprocessing  import check_valid_file
from preprocessing  import check_valid_dir

import numpy as np

from progressbar    import progressbar
from tqdm           import tqdm

import pyemblib
import sympy
import scipy
import time
import sys
import os 

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: a tuple of the script arguments
def parse_args():

    in_path = sys.argv[1]
    out_path = sys.argv[2]
    transform = sys.argv[3]

    args = [in_path,
            out_path,
            transform
            ]

    return args

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: translated matrix. 
def translation(matrix, args):

    direction, size = args

    # TRANSLATION

    # Generate the translation unit vector. 
    translation_vec = direction
 
    translation_size = size
    trans_matrix = []

    # Multiply our unit direction vector by our chosen scale factor. 
    translation_vec = translation_size * translation_vec   
 
    # Apply the translation. 
    for i,vector in tqdm(enumerate(matrix)):
        trans_matrix.append(matrix[i] + translation_vec)
        sys.stdout.flush()
    return trans_matrix

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: Homothetic transform of matrix. 
def homothetic(matrix, args):

    center, dilation_size = args

    # HOMOGENEOUS DILATION
    trans_matrix = []

    for i,vector in tqdm(enumerate(matrix)):
        center_diff = matrix[i] - center
        scaled_center_diff = dilation_size * center_diff
        trans_matrix.append(center + scaled_center_diff)
        sys.stdout.flush()
    
    return trans_matrix

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: Uniform scale of matrix. 
def uniform_scale(matrix, args):

    magnitude = args[0]

    # UNIFORM SCALE    
    
    trans_matrix = []

    for i,vector in tqdm(enumerate(matrix)):
        trans_matrix.append(magnitude * matrix[i])
        sys.stdout.flush()
    
    return trans_matrix

#========1=========2=========3=========4=========5=========6=========7==

# Note that a hyperplane through the origin in R^n is defined by a 
# single vector (hyperplane orthogonal to a). 
# RETURNS: Uniform scale of matrix. 
def reflect(matrix, args):

    hyperplane_vec = args[0]

    # REFLECTION        
    a = hyperplane_vec    
    trans_matrix = []

    for i,vector in tqdm(enumerate(matrix)):
        v = matrix[i]
        reflected_v = v - ((2 * np.dot(v, a) / np.dot(a, a)) * a)
        trans_matrix.append(reflected_v)
        sys.stdout.flush()
    
    return trans_matrix

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: Rotation of angle theta within the plane specified by
# the two vectors u,v.  
def rotate_2D(matrix, args):

    u,v,theta = args

    # PLANAR ROTATION IN R^n
    mat = []
    num_rows = len(matrix)
 
    dim = len(matrix[0])
    I = np.identity(dim)
    u = np.array([u])
    v = np.array([v])
    diff_1 = np.multiply(np.array(v),np.transpose(u)) - np.multiply(np.array(u),np.transpose(v))
    diff_2 = np.multiply(u,np.transpose(u)) - np.multiply(v,np.transpose(v))
    print("Following two shapes should be", str(dim), "x", str(dim))
    print(diff_1.shape)
    print(diff_2.shape)
    sys.stdout.flush()
    summand_1 = np.sin(theta) * diff_1
    summand_2 = (np.cos(theta)  - 1) * diff_2
    mat = I + summand_1 + summand_2
    
    rot_vectors = []

    rot_matrix = np.dot(matrix, mat)
    
    '''
    # There's gotta be a quicker way. 
    for i, row in tqdm(enumerate(matrix)):
        rot_row = np.multiply(row, mat)
        print("rot_row shape", rot_row)
        rot_vectors.append(rot_row)
        sys.stdout.flush()

    rot_matrix = np.array(rot_vectors)
    '''    

    # Should be (len, dim), is (len, dim, dim): fix it. 
    print("return shape: ",rot_matrix.shape)
    return rot_matrix

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: Shear mapping of something into something else. 
# PARAMETERS: W = [] list of vectors which span a subspace.
#             M = [] matrix representing a linear transformation from W to W'.
#                    Should be an k by (n - k) matrix where k is the dimension of W.  

def shear(W, M, theta, matrix):
    return 

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: transformed matrix. 

def transflow():
     
    check_valid_file(emb_path)
    if os.path.isfile(out_path):
        print("There is already a matrix saved with this name. ")
        exit() 

    # Take the first $n$ most frequent word vectors for a subset. 
    # Set to 0 to take entire embedding. 
    first_n = 0
   
    vectors_matrix,label_df = process_embedding(emb_path,first_n,None)
    num_rows = len(vectors_matrix)

    sample_vector = vectors_matrix[0]
    dimensions = len(sample_vector)


