"""
/*	Connor, Auburn, Turner [CAT]
 *	Set-Like operations on Text for Summaries
 *
 *
 */
"""
import json
from math import inf
from nltk.tokenize import sent_tokenize

DATA_FILE_NAME = "ProjectData.txt"

# Speed up testing by using only this many articles (inf -> all)
ARTICLE_LIMIT = 5

def main():
	print("Reading and Formatting Data")
	data = read_data()

	print(data)
	

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

def format(text):
	text = text.replace("\n", " ")
	text = text.replace("\'", "'")
	text = sent_tokenize(text)

	return text

if __name__ == "__main__":
	main()