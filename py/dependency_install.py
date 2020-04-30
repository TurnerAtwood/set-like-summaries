# Quick fix to install all of the dependencies for the package
import os

DEPENDENCIES = ['scipy', 'cython', 'numpy', 'sister', 'nltk', 'rouge', 'progressbar']

def main():
	for dep in DEPENDENCIES:
		os.system(f'pip install {dep}')

	import nltk
	nltk.download('punkt')

if __name__ == "__main__":
	main()


