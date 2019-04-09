from preprocessing import process_embedding
from preprocessing import check_valid_file
from preprocessing import check_valid_dir

import multiprocessing as mp
import tensorflow as tf 
import pandas as pd
import numpy as np

from progressbar import progressbar
from tqdm import tqdm

import pyemblib
import scipy
import queue
import time
import sys
import os 

#========1=========2=========3=========4=========5=========6=========7== 

# NEXTBATCH FUNCTION
'''
Function which creates a new batch of size batch_size, randomly chosen
from our dataset. For batch_size = 1, we are just taking one 100-dimen-
sional vector and computing its distance from every other vector in 
the dataset and then we have a num_inputs-dimensional vector which rep
-resents the distance of every vector from our "batch" vector. If we 
choose batch_size = k, then we would have k num_inputs-dimensional ve-
ctors. 
'''

def next_batch(entire_embedding,emb_transpose,label_df,
               batch_size,seed_queue,batch_queue):

    num_dimensions = int(entire_embedding.shape[1])
    name = mp.current_process().name
    print(name, 'Starting')
    sys.stdout.flush()
    with tf.Session() as sess: 
    
        print('TensorFlow session started successfully. ')
        sys.stdout.flush()
        
        slice_shape = [batch_size, num_dimensions]
 
        # Note slice_begin looks like "[row_loc, column_loc]", it is 
        # simply the coordinates of where we start our slice, so we 
        # set its placeholder to have shape(1,2)
        '''
        SLICE_BEGIN = tf.placeholder(tf.int32, shape=(2))
        slice_embedding = tf.slice(entire_embedding, 
                                   SLICE_BEGIN, slice_shape)
        '''       

        # This is a placeholder for the output of the "slice_embedding"
        # operation. It outputs a slice of the embedding, with 
        # shape "slice_shape". 
        SLICE_OUTPUT = tf.placeholder(tf.float32,shape=slice_shape)
        mult = tf.matmul(SLICE_OUTPUT,emb_transpose)

        # just need a value for "iteration" that is not -1 to satisfy
        # while condition on first loop
        iteration = 0 
        
        while True:
            while batch_queue.qsize() > 10:
                time.sleep(1)
                # print("Queue size is more than 10, waiting. ")       
           
            # print("grabbing a seed.") 
            sys.stdout.flush()
            iteration = seed_queue.get()
            # print("Iteration: ", iteration) 
            sys.stdout.flush()
            
            if iteration == -1:
                break

            current_index = iteration * batch_size 
            dist_row_list = []
    
            # get the corresponding slice of the labels as df
            slice_df = label_df[current_index:
                                current_index + batch_size]
            # slice_df = pd.DataFrame([0,0])
            # begin the slice at the "current_index"-th row in
            # the first column
            slice_begin = [current_index, 0]
        
            # slice the embedding from "slice_begin" with shape
            # "slice_shape"

            # TODO convert numpy slice
            slice_output = entire_embedding[current_index:current_index + batch_size,:num_dimensions]
            '''
            slice_output = sess.run(slice_embedding, 
                                    feed_dict={
                                     SLICE_BEGIN:slice_begin
                                    }
                                   )
            '''          

            # take dot product of slice with embedding
            dist_matrix = sess.run(mult, 
                                   feed_dict={
                                    SLICE_OUTPUT:slice_output
                                   }
                                  ) 
            
            # dist_matrix has shape 
            batch_queue.put([dist_matrix,slice_df])
            # print("pushed batch",iteration)
            sys.stdout.flush()
       
        # send halt 
        batch_queue.put([-1,-1])
    print(name, 'Exiting')
    sys.stdout.flush()
    return

#========1=========2=========3=========4=========5=========6=========7==
