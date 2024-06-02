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
    most_common_words = word_counts.most_common(5)

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


 ## need to remove punctation, stop words, and duplicate
url = 'https://en.wikipedia.org/wiki/Alan_Turing'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
paragraphs = soup.find_all('p')
data = []
for paragraph in paragraphs:
    data.append(paragraph.text)
data_new = ' '.join(data)
words_new = re.findall(r'\b\w+\b', data_new.lower())
# Tokenization
print_word_statistics(words_new, "Alan Turing")
tokens_all = tokenize(data_new)
stopwords = nltk.corpus.stopwords.words('english')
filtered_tokens = [token.lower() for token in tokens_all if token.lower() not in stopwords and token.isalpha()]
print_word_statistics(filtered_tokens, "Alan Turing After Tokenize")
filtered_tokens = list(set(filtered_tokens))
lemmas_all = lemmatize(filtered_tokens)
stems_all = stem(filtered_tokens)


print_word_statistics(lemmas_all, "Alan Turing After lemmatization")
print_word_statistics(stems_all, "Alan Turing After stem")
