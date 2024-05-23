import pandas as pd
data = pd.read_csv('spam.csv', encoding="latin1")

data = data.iloc[:, :-3]

print(data.head())

corpus = data.values.tolist()

for row in corpus:
    print(row)