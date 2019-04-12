# Nearest Neighbor Encoding

Implementation of Nearest Neighbor Encoding methods.

This library contains two components:

1. Nearest neighbor calculation
  - Scripts: `nn_saver.py`, `nearest_neighbors.py`
  - Implemented in Tensorflow
  - Uses cosine similarity to identify nearest neighbors
2. Graph generation
  - Script: `generate_graph.py`
  - Generates a weighted, directed edgelist file compatible with [node2vec](https://github.com/aditya-grover/node2vec)
