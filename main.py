
corpus = []
with open('spam.csv', 'r') as f:
    for line in f:
        corpus.append(line.strip())
