import numpy as np
"hello something the red"
["hello", "something", "the", "red"]
[10, 0, 20, 30]

def get_embedding_index(file_path):
    embeddings_index = {}
    f = open(file_path)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')  # 0 is the word
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index


def get_embedding_matrix(token_index_mapping, embedding_dimensions, embedding_index):
    num_tokens = len(token_index_mapping) + 1  # extra for for tokens not in vocablary
    embedding_matrix = np.zeros(shape=(num_tokens, embedding_dimensions))

    for word, index in token_index_mapping.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:  # careful with index 0
            embedding_matrix[index] = embedding_vector

    return embedding_matrix
