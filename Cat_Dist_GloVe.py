
#       Computes distance among vectors based on the    #
#       categorical distance computed as in the         #
#                   GloVe algorithm                     #



from tqdm import tqdm
import numpy as np

from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances

from corpus_raws import corpus_processing

corpus = corpus_processing('brown')

window_size = 2
from collections import defaultdict

# Create co-occurrence matrix
co_occurrence_matrix = defaultdict(lambda: defaultdict(int=0))

def compute_context_words(corpus: list[str], word_position: int, target_word: str, window_size: int=2) -> list[str]:
    start = max(0, word_position - window_size)
    end = min(len(corpus), word_position + window_size + 1)
    context_words = corpus[start:end]
    context_words.remove(target_word)
    return context_words

def compute_co_occurrence_matrix(corpus: list[str], window_size: int):
    _co_occurrence_matrix = defaultdict(lambda: defaultdict(int))
    for _i, _target_word in tqdm(enumerate(corpus), desc = "Computing GloVe Co-occurrence Matrix"):
        _context_words = compute_context_words(corpus, _i, _target_word, window_size)
        for _context_word in _context_words:
            _co_occurrence_matrix[_target_word][_context_word] += 1
    return _co_occurrence_matrix

def defaultdict_to_ndarray(co_occurrence_matrix: defaultdict):

    # Create a mapping from string keys to integer indices
    key_to_index = {key: idx for idx, key in enumerate(sorted(set(key for inner_dict in co_occurrence_matrix.values() for key in inner_dict.keys())))}

    # Determine the size of the ndarray
    num_rows = len(co_occurrence_matrix)
    num_cols = len(key_to_index)

    # Populate the ndarray
    co_occurrence_ndarray = np.zeros((num_rows, num_cols), dtype=int)

    for i, (key, inner_dict) in enumerate(co_occurrence_matrix.items()):
        for col_key, value in inner_dict.items():
            col_index = key_to_index[col_key]
            co_occurrence_ndarray[i, col_index] = value

    return co_occurrence_ndarray

def compute_GloVe_probability_matrix(co_occurrence_array: np.ndarray):
    row_norms = np.linalg.norm(co_occurrence_array, ord=1, axis=1, keepdims=True)
    GloVe_probability_array = co_occurrence_array/row_norms
    return GloVe_probability_array

def smoothing_probabilities(probability_array: np.array, type: str="laplace", alpha: float=1) -> np.ndarray:
    """
    Apply smoothings to a probability matrix.
    At the moment on Laplace smoothing with variable alpha.
    
    Parameters:
    - probability_array: np.array, theptobability matrix
    - type: str, the type of smoothing to apply (default is Laplace)
    - alpha: float, Laplace smoothing parameter (default is 1)
    
    Returns:
    - smoothed_probs: numpy.ndarray, the smoothed probability matrix
    """

    if type == "laplace":
        n = probability_array.shape[0]  # Number of rows or columns in the matrix
        smoothed_probs = (probability_array + alpha) / (np.sum(probability_array, axis=1, keepdims=True) + alpha * n)
    

    return smoothed_probs

def probability_to_dissimilarity(probability_array: np.array, alpha: float=1, beta: float=1, method: str="mds", Max: int=10000) -> np.ndarray:
    
    """
    Computes the dissimilarity matrix based on a probabililty matrix
    
    Parameters:
    - probability_array: np.array, the probability matrix.
    - alpha: float, first parameter of the linear transformation dij=alpha -beta*pij.
    - beta: float, what multiplies the probabilities, second parameter of the linear transformation dij=alpha -beta*pij.
    - method: str, string to select how to compute the dissimilarity matrix. Accepts parameters as "mds", "inverse" and "jaccard".
    - Max: int, the maximum dissimilarity value to replace nan's and inf in thedissimilarity matrix .
    
    Returns:
    - dissimilarity_matrix: numpy.ndarray, the smoothed probability matrix.
    """

    if method == "mds":
        dissimilarity_matrix = alpha - beta*probability_array
    elif method == "inverse":
        dissimilarity_matrix = 1 / probability_array
        # Replace infinite values with a large value (e.g., 1000)
        dissimilarity_matrix[np.isinf(dissimilarity_matrix)] = Max
    elif method == "jaccard":
        dissimilarity_matrix = pairwise_distances(probability_array, metric='jaccard')


    return dissimilarity_matrix


def probability_to_distance(probability_array: np.array):
    #direct computation prob -- [-log] --> dist
    #if d_{ij} = nan the elements have no relation
    distance_array = - np.log(probability_array)
    return distance_array



def mds_word_embedding(dimension: int, array: np.array, metric: bool, GloVe_metric):
        #---Maybe reference the paper for inspiration---#
    #[X] prepare the matrix to be compatible with the mds function:
        #[X] Diagonal has to be a certain way
        #[X] How to deal with the nan values
    #[X] Select mode := metric o no
    #[X] Select dimension of the embedding
    #[X] See what the output is

    if np.transpose(array) != array & GloVe_metric==False:
        array = np.dot(array,array.T)
    elif np.transpose(array) != array & GloVe_metric==True:
        array = (array+array.T)/2

    if metric == True and np.isnan(array).any() == True:
        print("This metric distance matrix has nan values and cannot be used with this settings")
        return -1

    mds = MDS(n_components=dimension, 
              metric=metric, 
              normalized_stress=True,   
              dissimilarity='precomputed'
              )
    mds_embedding = mds.fit_transform(array)
    stress = mds.stress_

    return mds_embedding, stress

#if __name__ == "__main__":



#library type --> typing information python
# scp file destination
#pypi pepy