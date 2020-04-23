from math import inf, ceil
from nltk.tokenize import sent_tokenize # Sentence tokenizer
from operator import itemgetter

import json
import numpy as np
import progressbar
import sister # Sentence Embedding generator
from rouge import Rouge
import sys

"""
COMP 5970 - Information Retrieval

Final Project
Set-Like Operations on Text for Summaries

Authors: Connor, Auburn, Turner [CAT]

The sky’s too fickle. It’s a play-place for butterflies.
"""

DATA_FILE_NAME = './static/txt/ProjectData.txt'
BIAS = {"1": "left-context", "2": "center-context", "3": "right-context"}
EMBEDDER = sister.MeanEmbedding(lang="en")

def summarize(article1, article2, operation_choice, num_sentences, their_summary):
  bias = {"r": 2, "c": 3, "l": 4}
  operation = {"difference": difference, "intersection": intersection, "union": union}
  article1_features = format(article1)
  article2_features = format(article2)
  our_summary = operation[operation_choice](article1_features, article2_features, num_sentences)
  scores = Rouge().get_scores(our_summary, their_summary)[0]

  return our_summary, scores


def intersection(article1, article2, num_sentences):
  indices = set_like_indices(article1, article2, num_sentences, True)
  return generate_summary(article1, indices)


def difference(article1, article2, num_sentences):
  indices = set_like_indices(article1, article2, num_sentences, False)
  return generate_summary(article1, indices)


def union(article1, article2, summary_percentage, num_sentences):
  return "Union operation not yet implemented."


# (operation) Intersection = True, Difference = False
# (summary_size_type) Sentence Number = False, Percentage Summary = True
# Returns set of indices from specified operation
def set_like_indices(article1, article2, num_sentences, operation):
  pairs = get_sentence_pairs(article1, article2)
  pairs.sort(key=itemgetter(0), reverse=operation)
  num_sentences = max(1, num_sentences)
  summary_size = min(len(article1), num_sentences)

  used_indices = [pair[1][2] for pair in pairs[:summary_size]]

  return used_indices


def generate_summary(article1, used_indices):
  ordered_indices = sorted(used_indices)

  return " ".join([article1[index][1] for index in ordered_indices])


# Pairs are [(cosine, v1, v2)]
# We only need the best pair for each sentence in a1
def get_sentence_pairs(article1, article2):
  best_pairs = list()
  for sentence1 in article1:
    best_score = -inf
    best_sentence = None
    for sentence2 in article2:
      score_with_sentence2 = cosine(sentence1[0], sentence2[0])
      if score_with_sentence2 > best_score:
        best_score = score_with_sentence2
        best_sentence = sentence2

    best_pairs.append((best_score, sentence1, best_sentence))

  return best_pairs


def cosine(vector1, vector2):
  return (vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))


# Convert the text to list of sentence feature tuples
# Sentence Features = [(embedding, sentence, index)]
def format(text):
  text = text.replace("\n", " ")
  text = text.replace("\'", "'")
  text = sent_tokenize(text)

  sentence_features = list()
  for index, sentence in enumerate(text):
    sentence_features.append( (EMBEDDER(sentence), sentence, index) )

  return sentence_features


def get_articles(topic_index, article1_bias, article2_bias):
  with open(DATA_FILE_NAME) as in_file:
    raw_data = json.load(in_file)
    article1 = raw_data['news%s' % topic_index][BIAS[article1_bias]]
    article2 = raw_data['news%s' % topic_index][BIAS[article2_bias]]
    their_summary = raw_data['news%s' % topic_index]['theme-description']

    return article1, article2, their_summary
