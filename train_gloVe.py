import numpy as np
import torch
from torch import nn
from torch.optim import Adam

from nltk.corpus import brown


# Preprocess the corpusfrom nltk.corpus import brown
from nltk.tokenize import sent_tokenize, word_tokenize

corpus = brown.sents()[:1]
corpus = [''.join(sent) for sent in corpus]
corpus = [word_tokenize(sent) for sent in corpus]

# Create a dictionary to store the vocabulary
vocabulary = {}
for sent in corpus:
    for word in sent:
        if word not in vocabulary:
            vocabulary[word] = len(vocabulary)
            
print(vocabulary)
# Compute co-occurrence counts
co_occurrence_counts = np.zeros((len(vocabulary), len(vocabulary)))
for sent in corpus:
    for i, word1 in enumerate(sent):
        for j in range(i + 1, len(sent)):
            word2 = sent[j]

            if word1 not in vocabulary or word2 not in vocabulary:
                continue

            co_occurrence_counts[vocabulary[word1], vocabulary[word2]] += 1

# Define the GloVe model
class GloVeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloVeModel, self).__init__()

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, x):
        return self.embeddings(x)

# Initialize the model
model = GloVeModel(len(vocabulary), 300)

# Train the GloVe model
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

for epoch in range(100):
    # Compute word vectors for each word pair
    word_vectors = model([[vocabulary.index(word1), vocabulary.index(word2)] for word1, word2 in vocabulary.items()])

    # Calculate the log-likelihood of each word pair
    log_likelihoods = torch.mm(word_vectors, word_vectors.t())

    # Calculate the negative log-likelihood loss
    loss = torch.mean(-log_likelihoods * torch.diag(co_occurrence_counts))

    # Backpropagate the loss and update the model parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

