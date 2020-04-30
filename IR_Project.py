from math import inf, ceil
from nltk.tokenize import sent_tokenize # Sentence tokenizer
from operator import itemgetter

import json
import numpy as np
import progressbar
import random
from rouge import Rouge
import sister # Sentence Embedding generator
import sys

"""
COMP 5970 - Information Retrieval

Final Project
Set-Like Operations on Text for Summaries

Authors: Connor, Auburn, Turner [CAT]

The sky’s too fickle. It’s a play-place for butterflies.
"""

# IDEA - Summarize articles before taking set operations
## This could remove redundant sentences so we don't have to deal with them

DATA_FILE_NAME = "ProjectData.txt"
TOPIC_LIMIT = inf # Speed up testing by using only this many articles (inf -> all)
EMBEDDER = sister.MeanEmbedding(lang="en")
BIAS_MAP = {"r": 2, "c": 3, "l": 4} # Maps specified bias to its index in the data
DEFAULT_RED_THRESH = 0.92

def main():
  data = read_data()

  run_interactive_mode(data)
  # run_test(data, "r", "l", 10, "1", "f")


def intersection(article1, article2, similarity, summary_size, summary_size_type, redundance_threshold):
  indices = set_like_indices(article1, article2, similarity, summary_size, summary_size_type, True, redundance_threshold)
  return generate_summary(article1, indices)


def difference(article1, article2, similarity, summary_size, summary_size_type, redundance_threshold):
  indices = set_like_indices(article1, article2, similarity, summary_size, summary_size_type, False, redundance_threshold)
  return generate_summary(article1, indices)


def union(article1, article2, similarity, summary_size, summary_size_type, redundance_threshold):
  return "Union operation not yet implemented."


# (operation) Intersection = True, Difference = False
# (summary_size_type) Sentence Number = False, Percentage Summary = True
# Returns set of indices from specified operation
def set_like_indices(article1, article2, similarity, summary_size, summary_size_type, operation, redundance_threshold):
  pairs = get_sentence_pairs(article1, article2, similarity)
  pairs.sort(key=itemgetter(0), reverse=operation)

  # Remove redundant sentences (from article1) in pairs
  pairs = remove_redundant_sentences(pairs, redundance_threshold)

  # This handles both % AND # summary specifications (but it is not pretty)
  if summary_size_type:
    summary_size = ceil(len(article1) * summary_size / 100)
  summary_size = max(0, summary_size)
  summary_size = min(len(pairs), summary_size)

  used_indices = [pair[1][2] for pair in pairs[:summary_size]]
  return used_indices


# Takes in pairs of sentences from 2 articles -> remove sentences
## from article1 that are too similar to each other
def remove_redundant_sentences(pairs, threshhold):
  kept_pairs = list()
  for i in range(len(pairs)):
    current_pair = pairs[i]

    current_pair_good = True
    for other_pair in kept_pairs:
      pairs_similarity = cosine(current_pair[1][0], other_pair[1][0])
      if pairs_similarity > threshhold:
        current_pair_good = False
        #print(f"\nBAD PAIR ({pairs_similarity}):\nREMOVED: {current_pair[1][1]}\nKEPT: {other_pair[1][1]}")
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


# Pairs are [(similarity, v1, v2)]
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
def format(text):
  text = text.replace("\n", " ")
  text = text.replace("\'", "'")
  text = sent_tokenize(text)

  sentence_features = list()
  for index, sentence in enumerate(text):
    sentence_features.append( (EMBEDDER(sentence), sentence, index) )

  return sentence_features


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
    formatted_topics[i] = (unformatted_topic['theme'],
                           unformatted_topic['theme-description'],
                           format(unformatted_topic['right-context']),
                           format(unformatted_topic['center-context']),
                           format(unformatted_topic['left-context']))
    bar.update(i + 1)
  bar.finish()

  return formatted_topics

def calc_scores(topics, nr_sentences, similarity, redundancy):
    red = DEFAULT_RED_THRESH
    if redundancy:
        red = 1
    scores = []

    count = 0
    for i,topic in enumerate(topics):
        if i % 50 == 0 and i != 0:
            print(f'{i} Topics Analyzed...')

        r_l_summ = intersection(topic[2],topic[4],similarity,nr_sentences,False,red)
        try:
            r_l_score = Rouge().get_scores(r_l_summ, topic[1])
        except:
            count += 1
            r_l_score = [{'rouge-1':{'f':0},'rouge-l':{'f':0}}]
        #l_c_score = Rouge().get_scores(intersection(topic[2], topic[3], similarity, nr_sentences, False, red), topic[1])
        l_r_summ = intersection(topic[4],topic[2],similarity,nr_sentences,False,red)
        try:
            l_r_score = Rouge().get_scores(l_r_summ, topic[1])
        except:
            count += 1
            r_l_score = [{'rouge-1':{'f':0},'rouge-l':{'f':0}}]
        #r_c_score = Rouge().get_scores(intersection(topic[4], topic[3], similarity, nr_sentences, False, red), topic[1])
        #c_r_score = Rouge().get_scores(intersection(topic[3], topic[4], similarity, nr_sentences, False, red), topic[1])
        #c_l_score = Rouge().get_scores(intersection(topic[3], topic[2], similarity, nr_sentences, False, red), topic[1])

        #scores.append((l_r_score,l_c_score,r_l_score,r_c_score,c_r_score,c_l_score))
        scores.append((l_r_score,r_l_score))
    print(f'{count} Fucking Explosions')
    return scores

# This is not needed in light of the web interface.
def run_interactive_mode(data):
  # List all topic themes (titles)
  title_output = []
  for i in range(len(data)):
    title_output.append(f"{i}: {data[i][0]}")
  print("\nTopic Themes:\n%s" % "\n".join(title_output))

  print('\nEntering main loop: (Example Input: 1 #/% 0/1 r i c C OR "new")\n')
  operation = {"d": difference, "i": intersection, "u": union}
  similarity = {"C": cosine, "E": euclidean, "M": manhattan}
  while(True):
    user_specs = input(">> Input (Enter to quit): ").split()

    if not user_specs:
      sys.exit("Exiting...")

    try:
      # The user will input his own articles
      if user_specs == ['new']:
        article1 = format(input(">> Enter the first article:\n"))
        article2 = format(input("\n>> Enter the second article:\n"))
        summary_size_type = bool(int(input("\n>> What type of summary? (0 - #, 1 - %): ")))
        summary_size = int(input("\n>> Summary Size (#/%): "))
        sim_choice = input("\n>> Similarity Function (C/E/M): ")
        oper_choice = input("\n>> Operation (d/i/u): ")
        summary = operation[oper_choice](article1, article2, similarity[sim_choice], summary_size, summary_size_type, DEFAULT_RED_THRESH)
        print(f"\nSummary:\n{summary}\n")

      # The user will use the existing articles
      else:
        topic_index = int(user_specs[0])
        summary_size = int(user_specs[1])
        summary_size_type = bool(int(user_specs[2]))
        article1 = data[topic_index][BIAS_MAP[user_specs[3]]]
        article2 = data[topic_index][BIAS_MAP[user_specs[5]]]
        oper_choice = user_specs[4]
        sim_choice = user_specs[6]

        summary = operation[oper_choice](article1, article2, similarity[sim_choice], summary_size, summary_size_type, DEFAULT_RED_THRESH)
        score = score_summary(summary, data[topic_index][1])
        print(f"\nOur Summary:\n{summary}\n")
        print(f"\nTheir Summary:\n{data[topic_index][1]}\n")
        print(f"Rouge Score: {score}")

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
