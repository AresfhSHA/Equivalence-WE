import nltk
from tqdm import tqdm
import re


def corpus_processing(corpus_name):

    corpus_dict = { #dictionary of the different corpora
        "brown": nltk.corpus.brown,
        "gutenberg": nltk.corpus.gutenberg,
        "reuters": nltk.corpus.reuters,
        "state_union": nltk.corpus.state_union,
        "treebank": nltk.corpus.treebank,
        "wordnet": nltk.corpus.wordnet,
        "movie_reviews": nltk.corpus.movie_reviews,
        "conll2000": nltk.corpus.conll2000,
        "conll2002": nltk.corpus.conll2002,
        "semcor": nltk.corpus.semcor,
        "floresta": nltk.corpus.floresta,
        "indian": nltk.corpus.indian,
        "mac_morpho": nltk.corpus.mac_morpho,
        "cess_cat": nltk.corpus.cess_cat,
        "cess_esp": nltk.corpus.cess_esp,
        "udhr": nltk.corpus.udhr,
        "tagged_treebank_para_block_reader": nltk.corpus.tagged_treebank_para_block_reader,
        "sinica_treebank": nltk.corpus.sinica_treebank,
        "alpino": nltk.corpus.alpino,
        "comparative_sentences": nltk.corpus.comparative_sentences,
        "dependency_treebank": nltk.corpus.dependency_treebank,
    }


    nltk.download(corpus_name)
    # Load the corpus
    # corpus_raw = nltk.corpus.brown.words()
    corpus_raw = corpus_dict[corpus_name].words()
    corpus_raw = [word.lower() for word in corpus_raw]
    # corpus = nltk.corpus.CorpusReader(nltk.data.find(f"corpora/{corpus_name}"), ".*")
    # corpus_raw = nltk.tokenize.word_tokenize(corpus.raw())
    print(f"Corpus size: {len(corpus_raw)}")
    corpus = []
    for word in tqdm(corpus_raw, desc="Pre-processing corpus"):
        if word is not None:
            if re.match(r'^[A-Za-z"\']', word):
                word = word.replace(",", "")
                corpus.append(word)

    return corpus_raw