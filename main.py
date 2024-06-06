import pandas as pd
import time
import nltk
import spacy
from nltk import PorterStemmer
from collections import Counter
import re
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
stopwords = nltk.corpus.stopwords.words('english')
filtered_words = [word for word in words if word not in stopwords and word not in string.punctuation]
filtered_string = ' '.join(filtered_words)


text = "can't do waiting doesnt going don't there's i've co-founded "
sample = nltk.word_tokenize(text)
sample2 = nlp(text)
print(sample)
for token in sample2:
    print(token.text, token.lemma_)

# NLTK TOKEN
start_time = time.time()
tokens_nltk = nltk.word_tokenize(filtered_string)
end_time = time.time()
print(f"It took {end_time - start_time:.2f} seconds to Tokenize with nltk")
print_word_statistics(tokens_nltk,'NLTK After Tokenize')
filtered_tokens2 = list(set(tokens_nltk))# no Dups


## SPACY TOKEN
start_time = time.time()
tokens_spacy = nlp(filtered_string)
end_time = time.time()
print(f"It took {end_time - start_time:.2f} seconds to Tokenize with spacy")
unique_tokens = {token.text: token for token in tokens_spacy if token.is_alpha}

# Get the list of unique token objects
unique_token_objects = list(unique_tokens.values())

# Create a set to store unique words
unique_words = set()
# Iterate through tokens
for token in tokens_spacy:
    # Filter out stopwords, punctuation, and spaces
    if not token.is_stop and not token.is_punct and not token.is_space:
        # Add the token's lowercase form to the set
        unique_words.add(token.text.lower())

# Get the number of unique words
filtered_tokens = [token.text.lower() for token in tokens_spacy]

# Count the frequency of each word
word_freq = Counter(filtered_tokens)

# Get the 5 most common words
most_common_words = word_freq.most_common(5)
num_unique_words = len(unique_words)
print(f'Number of Tokens :{len(tokens_spacy)}')
print(f"Number of unique words: {num_unique_words}")
print(f'Most 5 :{most_common_words}\n')





#lem With nltk
start_time = time.time()
lemmatizer = nltk.WordNetLemmatizer()
lemmatizer_words = [lemmatizer.lemmatize(word) for word in filtered_tokens2]
end_time = time.time()
print(f"It took {end_time - start_time:.2f} seconds to Lemmatize in nltk")
print_word_statistics(lemmatizer_words,'Nltk After Lemmatization')






## Lem With Spacy
start_time = time.time()
lemmatizer_words_spacy = [token.lemma_ for token in unique_token_objects]
end_time = time.time()
print(f"It took {end_time - start_time:.2f} seconds to lemmatize with spaCy")
print_word_statistics(lemmatizer_words_spacy,'Spacy After Lem')






# Stem in nltk
start_time = time.time()
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens2]
end_time = time.time()
print(f"It took {end_time - start_time:.2f} seconds to Stem in nltk")
print_word_statistics(stemmed_words,'NLTK in stem')





#Stem in spacy
start_time = time.time()
# Stem each token using NLTK's PorterStemmer
stemmed_tokens = [stemmer.stem(token.text) for token in unique_token_objects]
end_time = time.time()
print(f"It took {end_time - start_time:.2f} seconds to Stem in spacy")
print_word_statistics(stemmed_tokens,'Spacy After Stemming')
    ## END OF TEXT PROCESSING