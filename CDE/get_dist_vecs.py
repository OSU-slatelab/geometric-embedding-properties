import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from preprocessing  import process_embedding
from preprocessing  import check_valid_file
from preprocessing  import check_valid_dir
from next_batch     import next_batch

import tensorflow.contrib.layers    as lays
import multiprocessing              as mp
import tensorflow                   as tf 
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

'''
get_dist_vecs.py

Script to compute distance vectors for a set of vocab words from 
a pretrained embedding. 
'''

#========1=========2=========3=========4=========5=========6=========7==

# RETURNS: a tuple of the script arguments
def parse_args():

    emb_path = sys.argv[1]
    #model_path = sys.argv[2]
    # batch_size = int(sys.argv[3])
    # epochs = int(sys.argv[4])
    # learning_rate = float(sys.argv[5])
    # keep_prob = float(sys.argv[6])
    # num_processes = int(sys.argv[7])
    # vocab_path = int(sys.argv[8])

    args = [emb_path,
            10, 
            50,
            0.001,
            0.5,
            3,
            "/homes/3/whitaker.213/similarity_test/wordsim_vocab.txt"]

    return args

#========1=========2=========3=========4=========5=========6=========7==

# TRAINING FUNCTION
def epoch(embedding_tensor,num_batches,step,batch_queue,train,
          loss,loss_vectors,hidden_layer,X,init,saver,model_path,
          new_emb_path,retrain,num_processes):
 
    name = mp.current_process().name
    print(name, 'Starting')
    sys.stdout.flush()
    with tf.Session() as sess:
         
        # initializes all the variables that have been created
        sess.run(init)
        
        # list of slices which compose the new embedding
        embedding_slices = []
        label_slices = []

        # just can't be -1
        batch = np.zeros((5,5))
        total_error = 0
        batches_completed = 0
        print("number of batches: ", num_batches)
        halts = 0        


        while True:

            batch_loss = 0
            
            # print("about to try to grab")
            sys.stdout.flush()
            batch,slice_df = batch_queue.get()
            
            # print(type(slice_df))
            # print(slice_df.shape)
            
            # break for halt batch
            # be careful not to check for np.array but for np.ndarray!
            if not isinstance(batch, np.ndarray):
                print("Found a halt batch. ")
                halts += 1
                if halts >= num_processes:
                    break
                else:
                    # skip to next iteration of while loop
                    continue
            
            print("Batches completed: ", batches_completed, end="\r") 
            batches_completed = batches_completed + 1
            sys.stdout.flush()

            embedding_slices.append(batch)

            # add the slice of labels that corresponds to the batch
            label_slices.append(slice_df)

        # makes dist_emb_array a 3-dimensional array 
        dist_emb_array = np.stack(embedding_slices)
        
        # concatenates the first dimension, so dist_emb_array has 
        # shape [<num_inputs>,<dimensions>]
        dist_emb_array = np.concatenate(dist_emb_array)

        # concatenates the list of pands Series containing the words
        # that correspond to the new vectors in "dist_emb_array"
        labels = np.concatenate(label_slices)
        print("labels shape: ", labels.shape)
        print("dist_emb_array shape: ", dist_emb_array.shape)
        
        # creates the emb dict
        dist_emb_dict = {}
        for i in tqdm(range(len(labels))):
            emb_array_row = dist_emb_array[i]
            dist_emb_dict.update({labels[i]:emb_array_row})

        # saves the embedding
        pyemblib.write(dist_emb_dict, 
                       new_emb_path, 
                       mode=pyemblib.Mode.Text)

    while not batch_queue.empty():
        try:
            batch_queue.get(timeout=0.001)
        except:
            pass
 
    print(name, 'Exiting')
    return

#=========1=========2=========3=========4=========5=========6=========7=

def mkproc(func, arguments):
    p = mp.Process(target=func, args=arguments)
    p.start()
    return p

#========1=========2=========3=========4=========5=========6=========7==

def genflow(emb_path,batch_size,epochs,
            learning_rate,keep_prob,num_processes,vocab_path):

    assert batch_size > 1    

    emb_format = pyemblib.Format.Word2Vec
    print_sleep_interval = 1
    
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H%M")
    source_name = os.path.splitext(os.path.basename(emb_path))[0]
    parent = os.path.abspath(os.path.join(emb_path, "../"))

    model_path = "../AE_models/" + source_name + ".ckpt"
    model_index_path = model_path + ".index"

    retrain = True

    check_valid_file(emb_path)
    if os.path.isfile(model_index_path):

        print("There is already a model saved with this name. ") 
        time.sleep(print_sleep_interval)           
        sys.stdout.flush()
        retrain = False
    else:
        print("No existing model, exiting now. ")
        time.sleep(3)
        exit()

    with open(vocab_path, "r") as source:
        vocab = source.read().split('\n')

    # Take the first $n$ most frequent word vectors for a subset. 
    # Set to 0 to take entire embedding. 
    # Set size of distance vector target 
    # (i.e. dimensionality of distance vectors). 
    first_n = 10000
   
    dist_target,useless_labels = process_embedding( emb_path,
                                                    emb_format, 
                                                    first_n,
                                                    None)
    
    vectors_matrix,label_df = process_embedding(emb_path,
                                                emb_format, 
                                                0,
                                                None)

    # We get the dimensions of the input dataset. 
    shape = vectors_matrix.shape
    print("Shape of embedding matrix: ", shape)
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()

    # number of rows in the embedding 
    num_inputs = shape[0]
    num_outputs = num_inputs 

    # dimensionality of the embedding file
    num_hidden = shape[1]

    print("Learning rate is: ",learning_rate)
    time.sleep(print_sleep_interval)               
    sys.stdout.flush()
    
    # probability of outputting nonzero value in dropout layer. So the 
    # input to the dropout layer goes to zero 1 - keep_prob of the time 
    print("Dropout layer keep_prob is: ", keep_prob)
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()


    # clears the default graph stack
    tf.reset_default_graph()

    # PLACEHOLDER
    # "tf.float32" just means the data type is an integer. The shape is 
    # in the form [<columns>,<rows>], and "None" means it can be any 
    # value. So this placeholder can have any number of rows, and must 
    # have "num_inputs" columns. 
    print("Initializing placeholder. ")
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
    X = tf.placeholder(tf.float32, shape=[None, num_inputs])

    # WEIGHTS
    print("Initializing weights. ")
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
    # we use a variance scaling initializer so that it is capable of 
    # adapting its scale to the shape of the weight tensors. 
    initializer = tf.variance_scaling_initializer()
    input_weights = tf.Variable(initializer([num_inputs, num_hidden]), 
                                dtype=tf.float32)
    output_weights = tf.Variable(initializer([num_hidden, num_outputs]), 
                                 dtype=tf.float32)

    # BIAS
    input_bias = tf.Variable(tf.zeros(num_hidden))
    output_bias = tf.Variable(tf.zeros(num_outputs))

    # ACTIVATION
    act_func = tf.nn.relu

    print("Initializing layers and defining loss function. ")
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()

    #===================================================================

    # LAYERS
    # the argument of act_func is a Tensor, and the variable 
    # "hidden_layer" itself is also a Tensor. This hidden layer is just 
    # going to compute the element-wise relu 

    hidden_layer = act_func(tf.matmul(X, input_weights) + input_bias)

    # With probability keep_prob, outputs the input element scaled up 
    # by 1 / keep_prob, otherwise outputs 0. The scaling is so that the 
    # expected sum is unchanged.
    dropout_layer = tf.nn.dropout(hidden_layer,keep_prob=keep_prob)
    output_layer = tf.matmul(dropout_layer,output_weights)+output_bias 

    # We define our loss function, minimize MSE
    loss_vectors = tf.abs(output_layer - X)
    reduce_mean = tf.reduce_mean(X)
    loss = tf.reduce_mean(tf.abs(output_layer - X))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    train = optimizer.minimize(loss)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # change vectors matrix to just the vocab
    vectors_matrix,label_df = process_embedding(emb_path,
                                                pyemblib.Format.Word2Vec, 
                                                first_n,
                                                vocab)

    # Reset dimensions for vocab subset
    shape = vectors_matrix.shape
    print("Shape of embedding matrix: ", shape)
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()

    # Reset
    num_inputs = shape[0]
    
    # HYPERPARAMETERS
    num_batches = num_inputs // batch_size # floor division
    print("Defining hyperparameters: ")
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
    print("Epochs: ", epochs)
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
    print("Batch size: ", batch_size)
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
    print("Number of batches: ", num_batches)            
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
    
    # UNIT NORM THE EMBEDDING
    print("Unit norming the embedding. ")
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
    norms_matrix = np.linalg.norm(vectors_matrix, axis=1)
    norms_matrix[norms_matrix==0] = 1
    vectors_matrix = vectors_matrix / np.expand_dims(norms_matrix, -1)
    print(vectors_matrix.shape)


    # we read the numpy array "vectors_matrix" into tf as a Tensor
    # embedding_tensor = tf.constant(vectors_matrix)
    dist_target_tensor = tf.constant(dist_target)

    # Not doing this anymore due to memory constraints. 
    embedding_tensor = vectors_matrix

    print("shape of emb_tens is: ", embedding_tensor.shape)
    time.sleep(print_sleep_interval) 
    sys.stdout.flush()
 
    embedding_unshuffled = np.copy(embedding_tensor)
    # emb_transpose_unshuf = np.transpose(embedding_unshuffled)
    # emb_transpose_unshuf = tf.cast(emb_transpose_unshuf, tf.float32)
    emb_transpose = tf.transpose(dist_target_tensor)
    emb_transpose = tf.cast(emb_transpose, tf.float32)

    #===================================================================
    
    # program hangs when I try to run from saved model    
    ''' 
    # Later, launch the model, use the saver to restore variables from 
    # disk, and do some work with the model.
    with tf.Session() as sess:
      
        # Restore variables from disk.
        saver.restore(sess, model_path)
        print("Model restored.")
        
    # Check the values of the variables
    print(embedding_tensor.shape)

    # hidden_out = hidden_layer.eval(feed_dict={X: })
    # for row in hidden_out:
        # print(row) 
    '''

    eval_batch_size = batch_size
    assert eval_batch_size > 1

    # HYPERPARAMETERS
    eval_num_batches = num_inputs // eval_batch_size # floor division
    print("Defining hyperparameters: ")
    print("Eval batch size: ", eval_batch_size)
    print("Number of batches: ", eval_num_batches)
    
    # we instantiate the queue
    seed2_queue = mp.Queue()  
    
    mananger = mp.Manager()
    batch2_queue = mananger.Queue()
 
    # So we need each Process to take from an input queue, and 
    # to output to an output queue. All 3 batch generation 
    # prcoesses will read from the same input queue, and what 
    # they will be reading is just an integer which corresponds 
    # to an iteration 
    for iteration in tqdm(range(eval_num_batches)):  
        seed2_queue.put(iteration)

    # put in "p" halt seeds to tell the processes when to end
    for i in range(num_processes):
        seed2_queue.put(-1)

    print("seed queue size: ", seed2_queue.qsize())

    # CREATE MATRIXMULT PROCESSES
    batch_args = (embedding_unshuffled,
                  emb_transpose,
                  label_df,
                  eval_batch_size,
                  seed2_queue,
                  batch2_queue)
    print("About to start the batch processes. ")
    allprocs = [mkproc(next_batch, batch_args) 
                for x in range(num_processes)]

    # the name of the embedding to save
    # something like "~/<path>/steve.txt"
    new_emb_path =  str(os.path.join(parent, "full_dist-" + "__source--" + source_name 
                        + "__" + "time--" + timestamp + ".txt"))

    # Saving embedding vectors file. 
    retrain = False

    step = 0

    # RUN THE TRAINING PROCESS
    eval_process = mp.Process(name="eval",
                               target=epoch,
                               args=(embedding_unshuffled,
                                     eval_num_batches,
                                     step,
                                     batch2_queue,
                                     train,
                                     loss,
                                     loss_vectors,
                                     hidden_layer,
                                     X,
                                     init,
                                     saver,
                                     model_path,
                                     new_emb_path,
                                     retrain,
                                     num_processes))
    eval_process.start()    

    print("queue is full. ")
       
    ''' 
    # join the processes, i.e. end them
    for process in allprocs:
        process.terminate()
    '''

    # join the processes, i.e. end them
    for process in allprocs:
        process.join()

    eval_process.join()

    return

#========1=========2=========3=========4=========5=========6=========7==

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here 
    
    args = parse_args()

    emb_path = args[0]
    batch_size = args[1]
    epochs = args[2]
    learning_rate = args[3]
    keep_prob = args[4]
    num_processes = args[5]
    vocab_path = args[6]
    
    genflow(emb_path,batch_size,epochs,
              learning_rate,keep_prob,num_processes,vocab_path) 


