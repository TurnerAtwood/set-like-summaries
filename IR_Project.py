"""
/*	Connor, Auburn, Turner [CAT]
 *	Set-Like operations on Text for Summaries
 *
 *
 */
"""
import json

DATA_FILE_NAME = "ProjectData.txt"

def main():
	data = read_data()
	print(data[1475])

def read_data():
	with open(DATA_FILE_NAME) as in_file:
		# Grab the JSON dictionary from the file
		raw_data = json.load(in_file)
		data_size = len(raw_data)
		
		# Convert to a list based
		formatted_data = [0 for _ in range(data_size)]
		for article_name in raw_data:
			# Format: "news1234" (0 to n-1)
			index = int(article_name[4:])
			formatted_data[index] = raw_data[article_name]

		# Convert the articles to 5-tuples
		## (Title, summary, left, center, right)

		for i in range(data_size):
			c_art = formatted_data[i]
			formatted_data[i] = (c_art['theme'], c_art['theme-description'], clean(c_art['right-context']), clean(c_art['center-context']), clean(c_art['left-context']))

	return formatted_data

# TEXT CLEANER
LETTERS = {chr(i) for i in range(ord('a'), ord('z')+1)}
NUMBERS = {chr(i) for i in range(ord('0'), ord('9')+1)}
VALID = LETTERS.union({" ", "-", "'"}).union(NUMBERS)

def clean(text):
	text = text.lower()
	
	# Remove all dashes not part of words
	text = text.replace("--", " ")
	text = text.replace(" -", " ")
	text = text.replace("- ", " ")
	text = list(text)

	# Remove all punctuation
	for index in range(len(text)):
		text_char = text[index]
		if text_char not in VALID:
			text[index] = " "

	# Split words on spaces and newlines
	text = "".join(text)
	text = text.split()

	# Remove any word that contains a number or apostraphe
	"""
	new_text = []
	BAD = NUMBERS.union({"'"})
	for word in text:
		if len(BAD.intersection(set(word))) == 0:
			new_text.append(word)
	text = new_text
	"""

	return text

if __name__ == "__main__":
	main()