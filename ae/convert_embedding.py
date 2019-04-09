import pyemblib
import sys
import os

'''
convert_embedding.py

Quick and dirty script to convert embeddings from text to binary or
vice-versa. 
'''

def parse_args():

    emb_path = sys.argv[1]
    dest_path = sys.argv[2]
    mode = sys.argv[3]          # 'txt' or 'bin'.

    return [emb_path,dest_path,mode]

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

def read_embedding(emb_path):

    print("READING. ")
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

    return embedding

#========1=========2=========3=========4=========5=========6=========7==

def main(emb_path, dest_path, mode):

    embedding = read_embedding(emb_path)
    if mode == "txt":
        pyemblib.write(embedding, dest_path, mode=pyemblib.Mode.Text)
    elif mode == "bin":
        pyemblib.write(embedding, dest_path, mode=pyemblib.Mode.Binary)
    else:
        print("Mode (third argument) must be \"txt\" or \"bin\".")

#========1=========2=========3=========4=========5=========6=========7==

if __name__ == "__main__":
    # stuff only to run when not called via 'import' here 
    
    args = parse_args()
    emb_path = args[0]
    dest_path = args[1]
    mode = args[2]

    main(emb_path, dest_path, mode)
