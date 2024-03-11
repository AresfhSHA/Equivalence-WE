import nltk
from nltk.corpus import brown
from nltk.tokenize import sent_tokenize, word_tokenize
import math
import torch

from tqdm import tqdm


from corpus_raws import corpus_processing

corpus = corpus_processing('brown')

window_size = 2
from collections import defaultdict

# Create co-occurrence matrix
co_occurrence_matrix = defaultdict(lambda: defaultdict(int))

def compute_co_occurence_matrix(corpus):
    for i, target_word in enumerate(corpus):
        start = max(0, i - window_size)
        end = min(len(corpus), i + window_size + 1)
        context_words = corpus[start:end]
        context_words.remove(target_word)
        for context_word in context_words:
            #print(target_word)
            #print(context_word)
            co_occurrence_matrix[target_word][context_word] += 1

#print(sorted(co_occurrence_matrix.items())[:10])
        
import numpy as np

# Optionally, convert the matrix to a numpy array for further analysis
unique_words = list(set(corpus))
matrix_size = len(unique_words)
matrix = np.zeros((matrix_size, matrix_size))

word_to_index = {word: i for i, word in enumerate(unique_words)}
#print(co_occurrence_matrix)

for target_word, context_words in tqdm(co_occurrence_matrix.items(), desc = " Co-occurrence matrix"):
    for context_word, count in context_words.items():
        matrix[word_to_index[target_word]][word_to_index[context_word]] = count

print("\nCo-occurrence matrix as a numpy array:")

# Set hyperparameters
embedding_size = 50
learning_rate = 0.1
epochs = 100

# Initialize word vectors
word_vectors = np.eye(len(unique_words), embedding_size)

word_vectors = torch.tensor(word_vectors, requires_grad=True)
learning_rate = 0.01
epochs = 25
batch_size = 64  # Adjust as needed

optimizer = torch.optim.SGD([word_vectors], lr=learning_rate)

for epoch in tqdm(range(epochs),desc = "Training loop"):
    total_loss = 0

    for target_word, context_words in tqdm(co_occurrence_matrix.items(), desc = f"Epoch {epoch + 1}"):
        for context_word, count in context_words.items():
            i = word_to_index[target_word]
            j = word_to_index[context_word]

            dot_product = torch.dot(word_vectors[i], word_vectors[j])
            log_co_occurrence = np.log(count)

            loss = (dot_product - log_co_occurrence)**2
            total_loss += loss.item()

            if (i + 1) % batch_size == 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    print(f"Epoch {epoch + 1}, Loss: {total_loss}")