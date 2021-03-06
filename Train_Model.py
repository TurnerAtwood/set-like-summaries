import json
import progressbar

from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from math import inf

STOPWORDS = set(stopwords.words('english'))

DATA_FILE_NAME = './static/txt/ProjectData.txt'
TOPIC_LIMIT = inf

def main():
  training_sentences = read_data()
  model = Word2Vec(training_sentences, min_count=1, workers=8)
  model.save('model.bin')

def read_data():
  with open(DATA_FILE_NAME) as in_file:
    # Grab the JSON dictionary from the file
    raw_topics = json.load(in_file)

  data_size = min(len(raw_topics), TOPIC_LIMIT)

  print(f"Reading data into a list of sentences ({data_size} topics)...")

  total_word_count = 0
  stop_word_count = 0

  # Dump into a list of sentences (list of words)
  all_sentences = list()

  bar = progressbar.ProgressBar(maxval=data_size, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
  bar.start()
  for i in range(data_size):
    topic = raw_topics[f"news{i}"]
    for item_title in topic:
      raw_text_item = topic[item_title]
      # Convert from paragraph to a list of sentences
      text_sentences = sent_tokenize(raw_text_item)
      for sentence in text_sentences:
        sentence = clean_sentence(sentence)
        if sentence:
          sentence_words = word_tokenize(sentence)
          total_word_count += len(sentence_words)

          sentence_words = remove_stop_words(sentence_words)

          stop_word_count += len(sentence_words)
          all_sentences.append(sentence_words)

    bar.update(i+1)
  bar.finish()

  print(total_word_count)
  print(stop_word_count)

  return all_sentences

def clean_sentence(text):
  text = text.strip()
  text = text.replace(".","")
  return text

def remove_stop_words(text):
  remaining_words = list()
  for word in text:
    if word not in STOPWORDS:
      remaining_words.append(word)
  return remaining_words

if __name__ == "__main__":
  main()