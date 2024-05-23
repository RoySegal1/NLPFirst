
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from bs4 import BeautifulSoup
import requests
from collections import Counter
import re

# Open the text file in read mode
with open('whatsup.txt', 'r',encoding='utf-8') as file:
    # Read the entire contents of the file into a string
    file_contents = file.read()

words = re.findall(r'\b\w+\b', file_contents.lower())

def print_word_statistics(words, title):
    word_counts = Counter(words)
    total_words = len(words)
    unique_words = len(word_counts)
    most_common_words = word_counts.most_common(50)

    print(f"{title} Statistics:")
    print(f"Total words: {total_words}")
    print(f"Unique words: {unique_words}")
    print(f"Most common words: {most_common_words}")
    print("\n")


def tokenize(text):
    return word_tokenize(text)


# Lemmatization

def lemmatize(tokens):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(token) for token in tokens]


# Stemming
def stem(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in tokens]

tokens = word_tokenize(file_contents)
lemmatized_tokens = lemmatize(tokens)
stemmed_tokens = stem(tokens)
print_word_statistics(words, "Roy")
print_word_statistics(tokens, "Roy - Token")
print_word_statistics(lemmatized_tokens, "Roy-Lemma")
print_word_statistics(stemmed_tokens, "Roy-Stem")
