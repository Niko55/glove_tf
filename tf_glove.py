import nltk
import numpy as np
import tensorflow as tf


EMBEDDING_DIMENSION=50   
PAD_TOKEN = 0

word2idx = { 'PAD': PAD_TOKEN } # dict so we can lookup indices for tokenising our text later from string to sequence of integers
weights = []

with open ('glove.6B.50d.txt', 'r', encoding="utf8") as file:
    for index, line in enumerate(file):
        values = line.split() # Word and weights separated by space
        word = values[0] # Word is first symbol on each line
        word_weights = np.asarray(values[1:], dtype=np.float32) # Remainder of line is weights for word
        word2idx[word] = index + 1 # PAD is our zeroth index so shift by one
        weights.append(word_weights)       
        if index + 1 == 40_000:
            # Limit vocabulary to top 40k terms
            break

EMBEDDING_DIMENSION = len(weights[0])
# Insert the PAD weights at index 0 now we know the embedding dimension
weights.insert(0, np.random.randn(EMBEDDING_DIMENSION))

# Append unknown and pad to end of vocab and initialize as random
UNKNOWN_TOKEN=len(weights)
word2idx['UNK'] = UNKNOWN_TOKEN
weights.append(np.random.randn(EMBEDDING_DIMENSION))

# Construct our final vocab
weights = np.asarray(weights, dtype=np.float32)

VOCAB_SIZE=weights.shape[0]

features = {}
features['word_indices'] = nltk.word_tokenize('hello world') # ['hello', 'world']
features['word_indices'] = [word2idx.get(word, UNKNOWN_TOKEN) for word in features['word_indices']]

glove_weights_initializer = tf.constant_initializer(weights)
embedding_weights = tf.get_variable(
    name='embedding_weights', 
    shape=(VOCAB_SIZE, EMBEDDING_DIMENSION), 
    initializer=glove_weights_initializer,
    trainable=False)
embedding = tf.nn.embedding_lookup(embedding_weights, features['word_indices'])
#print(embedding)
