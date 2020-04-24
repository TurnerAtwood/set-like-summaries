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

# IDEA - Summarize articles before taking set operations
## This could remove redundant sentences so we don't have to deal with them

DATA_FILE_NAME = "ProjectData.txt"
TOPIC_LIMIT = 30 # Speed up testing by using only this many articles (inf -> all)
EMBEDDER = sister.MeanEmbedding(lang="en")
BIAS_MAP = {"r": 2, "c": 3, "l": 4} # Maps specified bias to its index in the data

def main():
  print("Reading and formatting data...")
  data = read_data()

  run_interactive_mode(data)
  # run_test(data, "r", "l", 10, "1", "f")
  

# DEBUG - Rename after finalization
## For now, this is aims to find the optimal sentence length in [1, upper_bound]
def run_test(data, bias1, bias2, upper_bound, rouge_type, metric):
  bias1_index = BIAS_MAP[bias1]
  bias2_index = BIAS_MAP[bias2]

  best_size_counts = [0 for i in range(upper_bound+1)]

  print("Running Tests...")
  bar = progressbar.ProgressBar(maxval=len(data), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
  bar.start()
  for i in range(len(data)):
    bar.update(i+1)
    topic = data[i]
    # Ignore all present topics that have no reference summary
    if not topic[1]:
      continue
    article1 = topic[bias1_index]
    article2 = topic[bias2_index]

    best_score = -inf
    best_size = 0
    for summary_size in range(1, upper_bound+1):
      this_summary = intersection(article1, article2, summary_size, 0)
      this_score = score_summary(this_summary, topic[1], rouge_type, metric)

      if this_score > best_score:
        best_score = this_score
        best_size = summary_size

    best_size_counts[best_size] += 1
  bar.finish()

  # Print Results
  print(f"Best sentence length counts (rouge-{rouge_type}, metric: {metric})")
  for i in range(1,upper_bound):
      print(f"{i}: {best_size_counts[i]}")

def intersection(article1, article2, summary_size, summary_size_type):
  indices = set_like_indices(article1, article2, summary_size, summary_size_type, True)
  return generate_summary(article1, indices)


def difference(article1, article2, summary_size, summary_size_type):
  indices = set_like_indices(article1, article2, summary_size, summary_size_type, False)
  return generate_summary(article1, indices)


def union(article1, article2, summary_size, summary_size_type):
  return "Union operation not yet implemented."


# (operation) Intersection = True, Difference = False
# (summary_size_type) Sentence Number = False, Percentage Summary = True
# Returns set of indices from specified operation
def set_like_indices(article1, article2, summary_size, summary_size_type, operation):
  pairs = get_sentence_pairs(article1, article2)
  pairs.sort(key=itemgetter(0), reverse=operation)

  # This handles both % AND # summary specifications (but it is not pretty)
  if summary_size_type:
    summary_size = ceil(len(article1) * summary_size / 100)
  summary_size = max(0, summary_size)
  summary_size = min(len(article1), summary_size)

  used_indices = [pair[1][2] for pair in pairs[:summary_size]]
  return used_indices


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


# Use this for testing on the dataset
def read_data():
  with open(DATA_FILE_NAME) as in_file:
    # Grab the JSON dictionary from the file
    raw_topics = json.load(in_file)

  # This is bad, remove it when the limit is no longer needed
  data_size = min(len(raw_topics), TOPIC_LIMIT)

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

def run_test(nr_sentences):
    global TOPIC_LIMIT
    TOPIC_LIMIT = inf
    all_topics = read_data()
    topics = []

    # Remove all topics lacking a summary
    for topic in all_topics:
        if topic[1] != '':
            topics.append(topic)

    print(f'{len(topics) Topics')

    scores = []
    count = 0
    for topic in topics 
        scores.append((Rouge().get_scores(intersection(topic[2], topic[3], nr_sentences, False),topics[1]),Rouge().get_scores(intersection(topic[3], topic[2], nr_sentences, False),topics[1])))
        count += 1
        if count % 50 == 0:
            print(f'{count} Topics Analyzed...')
    return scores

# This is not needed in light of the web interface.
def run_interactive_mode(data):
  # List all topic themes (titles)
  title_output = []
  for i in range(len(data)):
    title_output.append(f"{i}: {data[i][0]}")
  print("\nTopic Themes:\n%s" % "\n".join(title_output))

  print('\nEntering main loop: (Example Input: 1 #/% 0/1 r i c OR "new")\n')
  operation = {"d": difference, "i": intersection, "u": union}
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
        oper_choice = input(">> Operation (d/i/u): ")
        summary = operation[oper_choice](article1, article2, summary_size, summary_size_type)
        print(f"\nSummary:\n{summary}\n")

      # The user will use the existing articles
      else:
        topic_index = int(user_specs[0])
        summary_size = int(user_specs[1])
        summary_size_type = bool(int(user_specs[2]))
        article1 = data[topic_index][BIAS_MAP[user_specs[3]]]
        article2 = data[topic_index][BIAS_MAP[user_specs[5]]]
        oper_choice = user_specs[4]
        
        summary = operation[oper_choice](article1, article2, summary_size, summary_size_type)
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
