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
# FILE_NAME = "TwoArticles.txt"
TOPIC_LIMIT = 5 # Speed up testing by using only this many articles (inf -> all)
EMBEDDER = sister.MeanEmbedding(lang="en")
PERCENT_TO_SUMMARIZE = 30

def main():
  print("Reading and formatting data...")
  data = read_data()

  # List all topic themes (titles)
  title_output = []
  for i in range(len(data)):
    title_output.append(f"{i}: {data[i][0]}")
  print("\nTopic Themes:\n%s" % "\n".join(title_output))

  print('\nEntering main loop: (Example Input: 1 #/% 0/1 r i c OR "new")\n')
  # DEBUG: Not sure about the name bias - maybe 'source_type' or 'source_bias'?
  bias = {"r": 2, "c": 3, "l": 4}
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
        article1 = data[topic_index][bias[user_specs[3]]]
        article2 = data[topic_index][bias[user_specs[5]]]
        oper_choice = user_specs[4]
        
        summary = operation[oper_choice](article1, article2, summary_size, summary_size_type)
        print(f"\nOwer Summary:\n{summary}\n")
        print(f"\nThey're Summary:\n{data[topic_index][1]}\n")
        scores = Rouge().get_scores(summary, data[topic_index][1])[0]
        for metric_name in scores:
          print(f"{metric_name}: {scores[metric_name]}")


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


def intersection(article1, article2, summary_percentage, summary_size_type):
  indices = set_like_indices(article1, article2, summary_percentage, summary_size_type, True)
  return generate_summary(article1, indices)


def difference(article1, article2, summary_percentage, summary_size_type):
  indices = set_like_indices(article1, article2, summary_percentage, summary_size_type, False)
  return generate_summary(article1, indices)


def union(article1, article2, summary_percentage, summary_size_type):
  return "Union operation not yet implemented."


# (operation) Intersection = True, Difference = False
# (summary_size_type) Sentence Number = False, Percentage Summary = True
# Returns set of indices from specified operation
def set_like_indices(article1, article2, summary_size, summary_size_type, operation):
  pairs = get_sentence_pairs(article1, article2)
  pairs.sort(key=itemgetter(0), reverse=operation)

  # DEBUG: Sorry for this - very ugly
  ## Do we want to allow 0?
  if summary_size_type:
    summary_size = ceil(len(article1) * summary_size / 100)
  summary_size = max(0, summary_size)
  summary_size = min(len(article1), summary_size)

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


"""
def read_data():
with open(FILE_NAME) as in_file:
# List with 2 articles
return [format(article) for article in json.load(in_file)]
"""


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


if __name__ == "__main__":
  main()
