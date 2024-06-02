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
import string


def print_word_statistics(words2, title):
    word_counts2 = Counter(words2)
    total_words = len(words2)
    unique_words2 = len(word_counts2)
    most_common_words2 = word_counts2.most_common(5)

    print(f"{title} Statistics:")
    print(f"Total words: {total_words}")
    print(f"Unique words: {unique_words2}")
    print(f"Most common words: {most_common_words2}")
    print("\n")


def remove_punctuation(input_string):
    # Define the translation table for removing punctuation
    translator = str.maketrans('', '', string.punctuation)
    return input_string.translate(translator)


data = pd.read_csv('spam.csv', encoding="latin1")
data = data.iloc[:, :-3]
nlp = spacy.load('en_core_web_sm')
print(data.head())
print(f"We have {len(data)} messages.")
# Extract the first column
label_counts = data.iloc[:, 0].value_counts()
print(f"We have {label_counts.get('ham', 0)} Ham and {label_counts.get('spam', 0)} Spam.")
# Extract the messages column
messages = data.iloc[:, 1]
word_counts = messages.apply(lambda x: len(x.split()))  ## gets the number of words
average_word_count = word_counts.mean()  ## gets the mean of the numbers of words
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


# NLTK TOKEN
start_time = time.time()
tokens_nltk = nltk.word_tokenize(all_words) # need to stop duplicate and stop words, think where to do each step
stopwords = nltk.corpus.stopwords.words('english')
filtered_tokens = [token.lower() for token in tokens_nltk if token.lower() not in stopwords and token.isalpha()]
end_time = time.time()
print(f"It took {end_time - start_time:.2f} seconds to Tokenize with nltk")
print_word_statistics(filtered_tokens,'NLTK After Tokenize')
filtered_tokens = list(set(filtered_tokens))# no Dups


## SPACY TOKEN
start_time = time.time()
tokens_spacy = nlp(all_words)
end_time = time.time()
print(f"It took {end_time - start_time:.2f} seconds to Tokenize with spacy")
tokens_without_stopwords = [token.text for token in tokens_spacy if not token.is_stop and token.text not in string.punctuation]
print_word_statistics(tokens_without_stopwords,'Spacy After Token')
tokens_without_stopwords = list(set(tokens_without_stopwords))



#lem With nltk
start_time = time.time()
lemmatizer = nltk.WordNetLemmatizer()
lemmatizer_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]
end_time = time.time()
print(f"It took {end_time - start_time:.2f} seconds to Lemmatize in nltk")
print_word_statistics(lemmatizer_words,'Nltk After Lemmatization')


start_time = time.time()
lemmatizer_words_spacy = [token.lemma_ for token in tokens_spacy if not token.is_stop and token.text not in string.punctuation]
end_time = time.time()
print(f"It took {end_time - start_time:.2f} seconds to lemmatize with spaCy")
print_word_statistics(lemmatizer_words_spacy,'Spacy After Lem')


# Stem in nltk
start_time = time.time()
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
end_time = time.time()
print(f"It took {end_time - start_time:.2f} seconds to Stem in nltk")
print_word_statistics(stemmed_words,'NLTK in stem')
    ## END OF TEXT PROCESSING