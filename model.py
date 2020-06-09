from gensim.models import Word2Vec,KeyedVectors
from typing import List
import numpy as np

def convert_model():
#    model = Word2Vec.load('model.bin')
#    model = KeyedVectors.load_word2vec_format('model.bin',binary=True)
    model = Word2VecEmbedding()
    return model

class Word2VecEmbedding(object):
    def __init__(self, lang: str="en") -> None:
        model = Word2Vec.load('model.bin')
        self.model = model

    def get_word_vector(self,word: str) -> np.ndarray:
        if word in self.model:
            return self.model[word]
        else:
            return np.random.rand(self.model.vector_size)

    def get_word_vectors(self, words: List[str]) -> np.ndarray:
        vectors = []
        for word in words:
            vectors.append(self.get_word_vector(word))
        return np.array(vectors)
