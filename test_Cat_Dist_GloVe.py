''' 
Preguntas
        1) Los tests se dejan siempre vacíos, ej: def test_compute():
        R) Se pueden pasar, pero hay que definirlos antes en'conftest.py'.
            Para definirlos hay que usar la sintaxis '@pytest.fixture'.

        2) No puedo definir cosas fuera de ellos?  
        R) Hay que hacerlo de forma específica usando un fichero a parte
            llamado 'conftest.py'.

'''
import numpy as np

from Cat_Dist_GloVe import (
    compute_context_words,
    compute_co_occurrence_matrix,
    defaultdict_to_ndarray,
    compute_GloVe_probability_matrix,
    probability_to_distance,
    smoothing_probabilities,
    mds_word_embedding,
    probability_to_dissimilarity,
)

# The Text should have:
#   [X] At least 2 sentences (maybe repeated)
#   [X] Some specific words to easily compute teh co-occurrence matrix
#   [X] Need to check if the edge-cases work properly



#context_words("tree") = {over: 2, the: 3 , quick: 1}
#context_words("the") = {over: 2, quick: 2 , brown: 2, jumps: 2,tree: 3, the: 2}
#context_words("quick") = {the: 2, brown: 2, fox: 2, tree: 1}

# paquete --> necesita un __init__.py para poder importar sin problemas

def test_compute_context_words():
    Text = "The quick brown fox jumps over the tree. The quick brown fox jumps over the tree."
    Text = Text.lower().replace(".","")
    tokens = Text.split()
    assert compute_context_words(tokens, 6, "the", 2) == ["jumps", "over", "tree", "the"]
    assert compute_context_words(tokens, 9,  "quick", 2) == ["tree", "the", "brown", "fox" ]
    assert compute_context_words(tokens, 7, "tree", 2) == ["over", "the", "the", "quick"]
    assert compute_context_words(tokens, 3, "fox", 2) == ["quick", "brown", "jumps", "over"]

def test_compute_co_occurrence_matrix():
    Text = "The quick brown fox jumps over the tree. The quick brown fox jumps over the tree."
    Text = Text.lower().replace(".","")
    tokens = Text.split()
    assert compute_co_occurrence_matrix(tokens, 2)["tree"] == {"over": 2, "the": 3 , "quick": 1}
    assert compute_co_occurrence_matrix(tokens, 2)["the"]["quick"] == 2
    assert compute_co_occurrence_matrix(tokens, 2)["the"] == {"over": 2, "quick": 2 , "brown": 2, "jumps": 2,"tree": 3, "the": 2}
    assert compute_co_occurrence_matrix(tokens, 2)["quick"] == {"the": 2, "brown": 2, "fox": 2, "tree": 1}

def test_defaultdict_to_ndarray():
    Text = "The quick brown fox jumps over the tree. The quick brown fox jumps over the tree."
    Text = Text.lower().replace(".","")
    tokens = Text.split()
    co_occurrence = compute_co_occurrence_matrix(tokens, 2)
    array = defaultdict_to_ndarray(co_occurrence)
    assert array[5,5] == 2.
    assert array[5,6] == 2.
    assert array[6,5] == 3.

def test_compute_GloVe_probability_matrix():
    Text = "The quick brown fox jumps over the tree. The quick brown fox jumps over the tree."
    Text = Text.lower().replace(".","")
    tokens = Text.split()
    co_occurrence = compute_co_occurrence_matrix(tokens, 2)
    array = defaultdict_to_ndarray(co_occurrence)
    prob_array = compute_GloVe_probability_matrix(array)

    assert prob_array[0,0] == 2/13
    assert prob_array[1,1] == 2/7
    assert prob_array[6,5] == 1/2

def test_probability_to_distance():
    Text = "The quick brown fox jumps over the tree. The quick brown fox jumps over the tree."
    Text = Text.lower().replace(".","")
    tokens = Text.split()
    co_occurrence = compute_co_occurrence_matrix(tokens, 2)
    array = defaultdict_to_ndarray(co_occurrence)
    prob_array = compute_GloVe_probability_matrix(array)
    dist_array = probability_to_distance(prob_array)

    assert dist_array[6,5] == -np.log(1/2)

    def test_mds_word_embedding():
       """ Text = "The quick brown fox jumps over the tree. The quick brown fox jumps over the tree."
        Text = Text.lower().replace(".","")
        tokens = Text.split()
        co_occurrence = compute_co_occurrence_matrix(tokens, 2)
        array = defaultdict_to_ndarray(co_occurrence)
        prob_array = compute_GloVe_probability_matrix(array)
        dist_array = probability_to_distance(prob_array)"""
       


