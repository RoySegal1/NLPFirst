import pandas as pd
import time
import nltk
import spacy
from nltk import PorterStemmer
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from collections import Counter
import re
from bs4 import BeautifulSoup
import requests
data = pd.read_csv('spam.csv', encoding="latin1")

data = data.iloc[:, :-3]
#nlp = spacy.load('en_core_web_sm')
print(data.head())
print(f"We have {len(data)} messages.")
# Extract the first column
label_counts = data.iloc[:, 0].value_counts()
print(f"We have {label_counts.get('ham',0)} Ham and {label_counts.get('spam',0)} Spam.")
# Extract the messages column
messages = data.iloc[:, 1]
word_counts = messages.apply(lambda x: len(x.split())) ## gets the number of words
average_word_count = word_counts.mean() ## gets the mean of the numbers of words
print(f"We have {average_word_count} words in average.")

# Convert all messages to a single string
all_words = ' '.join(messages)

# Remove any non-alphabetic characters and split into words
words = re.findall(r'\b\w+\b', all_words.lower())
# Count the occurrences of each word
word_counts = Counter(words)
# Get the 5 most common words
most_common_words = word_counts.most_common(5)
# Get the words that appear only once
words_appear_once = [word for word, count in word_counts.items() if count == 1]
# Print the 5 most frequent words
print("The 5 most frequent words are:")
for word, count in most_common_words:
    print(f"{word}: {count}")
print(f"Words that appear only once:{len(words_appear_once)}")

start_time = time.time()
tokens_nltk = nltk.word_tokenize(all_words)
end_time = time.time()
print(f"It took {end_time - start_time:.2f} seconds to Tokenize with nltk")
token_counts = Counter(tokens_nltk)
most_common_words = token_counts.most_common(5)
print(f"There Are {len(tokens_nltk)} Words after tokenization")
print("The 5 most frequent words after tokenization in nltk are")
for word, count in most_common_words:
    print(f"{word}: {count}")

# remove all duplicates in order to see the difference of lemmatization and tokenization
tokens_nltk_unique = list(set(tokens_nltk))
# start_time = time.time()
# tokens_spacy = [token.text for token in nlp(all_words)]
# end_time = time.time()
# print(f"It took {end_time - start_time:.2f} seconds to Tokenize with spacy")

start_time = time.time()
lemmatizer = nltk.WordNetLemmatizer()
lemmatizer_words = [lemmatizer.lemmatize(word) for word in tokens_nltk_unique]
end_time = time.time()
print(f"It took {end_time - start_time:.2f} seconds to Lemmatize in nltk")
lemmatizer_count = Counter(lemmatizer_words)
most_common_words = lemmatizer_count.most_common(5)
print(f"There Are {len(lemmatizer_words)} Words after Lemmatize")
print("The 5 most frequent words after Lemmatize in nltk are")
for word, count in most_common_words :
    print(f"{word}: {count}")

# # Lemmatization with spaCy
# start_time = time.time()
# # Create a spaCy document
# doc_spacy = nlp(all_words)
# # Extract lemmas
# lemmatizer_words_spacy = [token.lemma_ for token in doc_spacy]
# end_time = time.time()
# print(f"It took {end_time - start_time:.2f} seconds to lemmatize with spaCy")

# Stem in nltk
start_time = time.time()
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in tokens_nltk_unique]
end_time = time.time()
print(f"It took {end_time - start_time:.2f} seconds to Stem in nltk")
stem_count = Counter(stemmed_words)
most_common_words = stem_count.most_common(5)
print(f"There Are {len(lemmatizer_words)} Words after Stem")
print("The 5 most frequent words after Stem in nltk are")
for word, count in most_common_words:
    print(f"{word}: {count}")

    ## END OF TEXT PROCESSING
url = "https://www.facebook.com/roy.segal.739"
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')
paragraphs = soup.find_all('p')
for paragraph in paragraphs:
    print(paragraph.text)




