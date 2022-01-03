import numpy as np

def load_glove_model(File, vocab):
    print("Loading Glove Model...")
    glove_model = {}
    with open(File,'r', encoding='utf-8') as f:
        for line in f:
          try:
            split_line = line.split()
            word = split_line[0]
            if word in vocab:
              embedding = np.array(split_line[1:], dtype=np.float32)
              glove_model[word] = embedding
          except ValueError :
            continue
    
    return glove_model