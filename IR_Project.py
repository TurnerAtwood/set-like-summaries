"""
/*	Connor, Auburn, Turner [CAT]
 *	Set-Like operations on Text for Summaries
 *
 *
 */
"""
import json
from math import inf,ceil
import numpy as np
# Sentence Tokenizer
from nltk.tokenize import sent_tokenize
# Sentence Embedding generator
import sister

# IDEA - Summarize articles before taking set operations
## This could remove redundant sentences so we don't have to deal with them

# DATA_FILE_NAME = "ProjectData.txt"
FILE_NAME = "TwoArticles.txt"

# Speed up testing by using only this many articles (inf -> all)
ARTICLE_LIMIT = 2
EMBEDDER = sister.MeanEmbedding(lang="en")
PERCENT_SUMMARIZED = 100

def main():
	print("Reading and Formatting Data")
	a1,a2 = read_data()

	print("INTERSECTION")
	print(intersection(a1,a2))
	print()
	# print(intersection(a2,a1))

	print("DIFFERENCE")
	print(difference(a1,a2))
	print()
	# print(difference(a2,a1))

def intersection(a1, a2, summary_percent = PERCENT_SUMMARIZED):
	indices = set_like_indices(a1, a2, summary_percent, True)
	return gen_summary(a1, indices)

# IDEA - Get best match for each sentence in a1, then take the worst-best pairs
def difference(a1, a2, summary_percent = PERCENT_SUMMARIZED):
	indices = set_like_indices(a1, a2, summary_percent, False)
	return gen_summary(a1, indices)

def union():
	pass

# Intersection = True, Difference = False
# Returns set of indices from specified operation
def set_like_indices(a1, a2, summary_percent, oper): 
	pairs = get_sentence_pairs(a1, a2)
	pairs.sort(reverse=oper)

	summary_size = ceil(len(a1) * summary_percent / 100)

	# NOTE - No longer a set
	used_indices = [pair[1][2] for pair in pairs[:summary_size]]

	return used_indices

def gen_summary(a1, used_indices):
	ordered_indices = sorted(used_indices)

	return " ".join([a1[ind][1] for ind in ordered_indices])


# Pairs are [(cosine, v1, v2)]
# We only need the best pair for each sentence in a1
def get_sentence_pairs(a1, a2):
	best_pairs = list()
	for e1 in a1:
		best_score = -inf
		best_sentence = None
		for e2 in a2:
			e2_score = cosine(e1[0],e2[0])
			if e2_score > best_score:
				best_score = e2_score
				best_sentence = e2 

		best_pairs.append( (best_score, e1, best_sentence) )
	return best_pairs

def cosine(v1, v2):
	return (v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# Embeddings = [(embedding, sentence, index)]
def format(text):
	text = text.replace("\n", " ")
	text = text.replace("\'", "'")
	text = sent_tokenize(text)

	embeddings = list()
	for index in range(len(text)):
		sentence = text[index]
		embeddings.append( (EMBEDDER(sentence), sentence, index) )
	return embeddings

def read_data():
	with open(FILE_NAME) as in_file:
		# List with 2 articles
		return [format(article) for article in json.load(in_file)]

if __name__ == "__main__":
	main()

"""
# Use this for testing on the dataset
def read_data():
	with open(DATA_FILE_NAME) as in_file:
		# Grab the JSON dictionary from the file
		raw_data = json.load(in_file)
		
		# This is bad, remove it when the limit is no longer needed
		data_size = min(len(raw_data), ARTICLE_LIMIT)
		
		# Convert to a list based
		formatted_data = [0 for _ in range(data_size)]
		for article_name in raw_data:
			# Format: "news1234" (0 to n-1)
			index = int(article_name[4:])
			
			if index < ARTICLE_LIMIT:
				formatted_data[index] = raw_data[article_name]

		# Convert the articles to 5-tuples
		## (Title, summary, left, center, right)
		for i in range(data_size):
			c_art = formatted_data[i]
			formatted_data[i] = (c_art['theme'], c_art['theme-description'], format(c_art['right-context']), format(c_art['center-context']), format(c_art['left-context']))

	return formatted_data
"""