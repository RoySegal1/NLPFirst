import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from bs4 import BeautifulSoup
import requests
from collections import Counter
import re
def print_word_statistics(words, title):
    word_counts = Counter(words)
    total_words = len(words)
    unique_words = len(word_counts)
    most_common_words = word_counts.most_common(10)

    print(f"{title} Statistics:", file=file)
    print(f"Total words: {total_words}", file=file)
    print(f"Unique words: {unique_words}", file=file)
    print(f"Most common words: {most_common_words}", file=file)
    print("\n", file=file)


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


# Open the text file in read mode
with open('whatsup.txt', 'r', encoding='utf-8') as file:
    # Read the entire contents of the file into a string
    file_contents = file.read()


file = open("output_whatsapp.txt",'w')
words_new = re.findall(r'\b\w+\b', file_contents.lower())
stopwords = nltk.corpus.stopwords.words('english')
words_new = [token.lower() for token in words_new if token.lower() not in stopwords and token.isalpha()]
tokens_new = word_tokenize(file_contents)
filtered_tokens = [token.lower() for token in tokens_new if token.lower() not in stopwords and token.isalpha()]
filtered_tokens = list(set(filtered_tokens))
lemmatized_tokens = lemmatize(filtered_tokens)
stemmed_tokens = stem(filtered_tokens)
print_word_statistics(words_new, "Roy")
print_word_statistics(tokens_new, "Roy - Token")
print_word_statistics(lemmatized_tokens, "Roy-Lemma")
print_word_statistics(stemmed_tokens, "Roy-Stem")
file.close()