from math import inf, ceil
from nltk.tokenize import sent_tokenize # Sentence tokenizer
from operator import itemgetter

import json
import numpy as np
import progressbar
from rouge import Rouge
import random
import sister # Sentence Embedding generator
import sys

"""
COMP 5970 - Information Retrieval

Final Project
Set-Like Operations on Text for Summaries

Authors: Connor, Auburn, Turner [CAT]

The sky’s too fickle. It’s a play-place for butterflies.
"""

DATA_FILE_NAME = './static/txt/ProjectData.txt'
EMBEDDER = sister.MeanEmbedding(lang="en")
TOPIC_LIMIT = inf # Set to inf to run over all topics in CLI

def main():
  data = read_data()
  run_interactive_mode(data)


def summarize(article1, article2, operation_choice, similarity_choice, num_sentences, their_summary):
  operation = {"difference": difference, "intersection": intersection}
  similarity = {"cosine": cosine, "euclidean": euclidean, "manhattan": manhattan}
  article1_features = get_sentence_features(article1)
  article2_features = get_sentence_features(article2)
  our_summary = operation[operation_choice](article1_features, article2_features, similarity[similarity_choice], num_sentences)
  scores = Rouge().get_scores(our_summary, their_summary)[0]

  return our_summary, scores


def intersection(article1, article2, similarity, num_sentences, redundance_threshold=0.92):
  indices = set_like_indices(article1, article2, similarity, num_sentences, True, redundance_threshold)
  return generate_summary(article1, indices)


def difference(article1, article2, similarity, num_sentences, redundance_threshold=0.92):
  indices = set_like_indices(article1, article2, similarity, num_sentences, False, redundance_threshold)
  return generate_summary(article1, indices)


# (operation) Intersection = True, Difference = False
# (summary_size_type) Sentence Number = False, Percentage Summary = True
# Returns set of indices from specified operation
def set_like_indices(article1, article2, similarity, num_sentences, operation, redundance_threshold):
  pairs = get_sentence_pairs(article1, article2, similarity)
  pairs.sort(key=itemgetter(0), reverse=operation)

  pairs = remove_redundant_sentences(pairs, redundance_threshold)

  num_sentences = max(1, num_sentences)
  summary_size = min(len(article1), num_sentences)

  used_indices = [pair[1][2] for pair in pairs[:summary_size]]

  return used_indices


def remove_redundant_sentences(pairs, threshold):
  kept_pairs = list()
  for i in range(len(pairs)):
    current_pair = pairs[i]

    current_pair_good = True
    for other_pair in kept_pairs:
      pairs_similarity = cosine(current_pair[1][0], other_pair[1][0])
      if pairs_similarity > threshold:
        current_pair_good = False
        # print(f"\nBAD PAIR ({pairs_similarity}):\n{current_pair[1][1]}\n{other_pair[1][1]}")
        break
    if current_pair_good:
      kept_pairs.append(current_pair)

  return kept_pairs


def generate_summary(article1, used_indices):
  ordered_indices = sorted(used_indices)
  return " ".join([article1[index][1] for index in ordered_indices])


# Wrapper for the rouge function to pick out a single rouge and metric.
## type = '1'/'2'/'l', metric = 'f'/'p'/'r'
def score_summary(hypothesis, reference, rouge_type = 'l', metric = 'f'):
  scores = Rouge().get_scores(hypothesis, reference)[0]
  rouge_l = scores[f'rouge-{rouge_type}'][metric]
  return rouge_l


# Pairs are [(cosine, v1, v2)]
# We only need the best pair for each sentence in a1
def get_sentence_pairs(article1, article2, similarity):
  best_pairs = list()
  for sentence1 in article1:
    best_score = -inf
    best_sentence = None
    for sentence2 in article2:
      score_with_sentence2 = similarity(sentence1[0], sentence2[0])
      if score_with_sentence2 > best_score:
        best_score = score_with_sentence2
        best_sentence = sentence2

    best_pairs.append((best_score, sentence1, best_sentence))

  return best_pairs


def cosine(vector1, vector2):
  return (vector1.dot(vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))


def euclidean(vector1, vector2):
  return np.linalg.norm(vector1 - vector2)


def manhattan(vector1, vector2):
  diffs = np.subtract(vector1, vector2)
  return sum([abs(diff) for diff in diffs])


def random_selection(vector1, vector2):
  return random.random()


# Convert the text to list of sentence feature tuples
# Sentence Features = [(embedding, sentence, index)]
def get_sentence_features(text):
  text = sent_tokenize(text)

  sentence_features = list()
  for index, sentence in enumerate(text):
    sentence_features.append( (EMBEDDER(sentence), sentence, index) )

  return sentence_features


def format(text):
  text = text.replace("\n", " ")
  text = text.replace("\'", "'")

  return text


def get_articles(topic_index, article1_bias, article2_bias):
  bias = {"1": "left-context", "2": "center-context", "3": "right-context"}

  with open(DATA_FILE_NAME) as in_file:
    raw_data = json.load(in_file)
    article1 = format(raw_data['news%s' % topic_index][bias[article1_bias]])
    article2 = format(raw_data['news%s' % topic_index][bias[article2_bias]])
    their_summary = raw_data['news%s' % topic_index]['theme-description']

    return article1, article2, their_summary


# Use this for testing on the dataset
def read_data():
  with open(DATA_FILE_NAME) as in_file:
    # Grab the JSON dictionary from the file
    raw_topics = json.load(in_file)

  # This is bad, remove it when the limit is no longer needed
  data_size = min(len(raw_topics), TOPIC_LIMIT)

  print(f"Reading and formatting data ({data_size} topics)...")


  # Convert to a list based
  formatted_topics = [0 for _ in range(data_size)]
  for topic_index in range(data_size):
    formatted_topics[topic_index] = raw_topics['news%d' % topic_index]

  bar = progressbar.ProgressBar(maxval=data_size, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

  # Convert the topics to 5-tuples
  # (Title, summary, right, center, left)
  bar.start()
  for i in range(data_size):
    unformatted_topic = formatted_topics[i]
    right = format(unformatted_topic['right-context'])
    center = format(unformatted_topic['center-context'])
    left = format(unformatted_topic['left-context'])
    formatted_topics[i] = (unformatted_topic['theme'],
                           unformatted_topic['theme-description'],
                           get_sentence_features(right),
                           get_sentence_features(center),
                           get_sentence_features(left))
    bar.update(i + 1)
  bar.finish()

  return formatted_topics


def calc_scores(topics, nr_sentences, similarity, redundancy):
  red = 0.92
  if redundancy:
    red = 1
  scores = []

  count = 0
  for i,topic in enumerate(topics):
    if i % 50 == 0 and i != 0:
      print(f'{i} Topics Analyzed...')

    r_l_summ = intersection(topic[2], topic[4], similarity, nr_sentences, red)
    try:
      r_l_score = Rouge().get_scores(r_l_summ, topic[1])
    except:
      count += 1
      r_l_score = [{'rouge-1': {'f':0}, 'rouge-l': {'f':0}}]
    #l_c_score = Rouge().get_scores(intersection(topic[2], topic[3], similarity, nr_sentences, red), topic[1])
    l_r_summ = intersection(topic[4], topic[2], similarity, nr_sentences, red)
    try:
      l_r_score = Rouge().get_scores(l_r_summ, topic[1])
    except:
      count += 1
      r_l_score = [{'rouge-1': {'f':0}, 'rouge-l': {'f':0}}]
    #r_c_score = Rouge().get_scores(intersection(topic[4], topic[3], similarity, nr_sentences, red), topic[1])
    #c_r_score = Rouge().get_scores(intersection(topic[3], topic[4], similarity, nr_sentences, red), topic[1])
    #c_l_score = Rouge().get_scores(intersection(topic[3], topic[2], similarity, nr_sentences, red), topic[1])

    #scores.append((l_r_score, l_c_score, r_l_score, r_c_score, c_r_score, c_l_score))
    scores.append((l_r_score, r_l_score))
  print(f'{count} Bad sentences')
  return scores


# A CLI based implementation.
def run_interactive_mode(data):
  # List all topic themes (titles)
  title_output = []
  for i in range(len(data)):
    title_output.append(f"{i}: {data[i][0]}")

  print("\nTopic Themes:\n%s" % "\n".join(title_output))
  print('\nEntering main loop: (Example Input: 1 # r i c C OR "new")\n')

  operation = {"d": difference, "i": intersection}
  similarity = {"C": cosine, "E": euclidean, "M": manhattan}
  bias = {"r": 2, "c": 3, "l": 4}

  while(True):
    user_specs = input(">> Input (Enter to quit): ").split()

    if not user_specs:
      sys.exit("Exiting...")

    try:
      # The user will input his own articles
      if user_specs == ['new']:
        article1 = format(input("\n>> Enter the first article:\n"))
        article2 = format(input("\n>> Enter the second article:\n"))
        their_summary = input("\n>> Enter your summary:\n")
        summary_size = int(input("\n>> Summary Size (#): "))
        oper_choice = input("\n>> Operation (d/i): ")
        sim_choice = input("\n>> Similarity Function (C/E/M): ")

        summary = operation[oper_choice](get_sentence_features(article1), get_sentence_features(article2), similarity[sim_choice], summary_size)
        score = score_summary(summary, their_summary)

      # The user will use the existing articles
      else:
        topic_index = int(user_specs[0])
        summary_size = int(user_specs[1])
        article1 = data[topic_index][bias[user_specs[2]]]
        article2 = data[topic_index][bias[user_specs[4]]]
        oper_choice = user_specs[3]
        sim_choice = user_specs[5]

        summary = operation[oper_choice](article1, article2, similarity[sim_choice], summary_size)
        score = score_summary(summary, data[topic_index][1])
        print(f"\nTheir Summary:\n{data[topic_index][1]}\n")
      print(f"\nOur Summary:\n{summary}\n")
      print(f"Rouge Score: {score}\n")

    # Constructive user error handling :)
    except KeyError as E:
      print("\nKeyError: %s" % E)
      print("Way to go idiot.\n")
    except ValueError as E:
      print("\nValueError: %s" % E)
      print("Really? Great job you fart.\n")
    except IndexError as E:
      print("\nIndexError: %s" % E)
      print("Not even close. Try reading more.\n")


if __name__ == "__main__":
  main()
