import pandas as pd

df = pd.read_pickle("books_with_embeddings.pkl")
print(df.head())
print("Embeddings for first book:", df.loc[0, "embedding"][:10], "...")  # first 10 values