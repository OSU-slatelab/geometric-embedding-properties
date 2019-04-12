# embedding-encoder
Autoencoder for word2vec word embedding files.

### train.sh

Main training script. Runs ae.py. 

### dist\_only.sh

Script to compute and save distance vectors for a given vocab set. 


### ae.py

Main script. Calls preprocessing.py and next\_batch.py. 
Given an embedding in .txt or .bin format, preproceses, and generates
in batches distance vectors. Uses single-hidden layer autoencoder to
compress distance vectors into shape of source embedding file. 
Saves the model and saves embedding vectors as text file. If the script
is run with a model name which already exists, it saves the embedding 
vectors instead of retraining.  

### next\_batch.py

Function which creates a new batch of size batch\_size, randomly chosen
from our dataset. For batch\_size = 1, we are just taking one 100-dimen-
sional vector and computing its distance from every other vector in 
the dataset and then we have a num\_inputs-dimensional vector which rep
-resents the distance of every vector from our "batch" vector. If we 
choose batch\_size = k, then we would have k num\_inputs-dimensional ve-
ctors.

### rand\_vecs.py 

Script to generate an embedding with random, normalized vectors for each
token from a source vocab file. 


### preprocessing.py

Preprocessing for ae.py


### convert\_embedding.py

Quick and dirty script to convert embeddings from text to binary or
vice-versa. 

### emb\_modulus.py 

Compute and print the average vector norm given the location of a
pretrained embedding.  

### save\_first\_n.py

Saves first n most frequent vectors given a pretrained embedding file. 







### nn.py

Unfinished script to compute nearest neighbors. 


### notes.txt 

Development notes. 







The ideas is that we pick one (ora  few, this is "batch\_size"), and compute the distance from this embedding to all others, and train on this at each step. 
A placeholder is a stand-in for our dataset. We'll assign data to it at a later date. Data is "fed" into the persistent TensorFlow network graph through these placeholders. 
