from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

'''sentences = [
    "The students are worried about the final projects",
    "The professor reviewed the final projects",
    "The class discussed their awful notes on the final projects",
]

# Tokenizar las oranciones (Corpus)
tokenized_sentences = [word_tokenize(sentence.lower()) for sentence in sentences]
print(f"Tokenized sentences: {tokenized_sentences}")


# Entrenar el modelo Word2Vec
model = Word2Vec(sentences=tokenized_sentences,
                 vector_size=10,
                 window=3,
                 min_count=1)

# Mostrar el vocabulario y los vectores de palabras
vocab = list(model.wv.key_to_index)
print(f"Vocab: {vocab}")

vectors = model.wv[vocab]
print(f"Vectors: {vectors}")
print()

print(f"Embedding for 'students': {model.wv['students']}")

# Reducir la dimensionalidad de los vectores de palabras
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vectors)
# print(f"2D vectors: {vectors_2d}")

plt.figure(figsize=(10, 8))
# Traemos todos los x y todos los y
plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], color='blue')

for i, word in enumerate(vocab):
    plt.annotate(word, [vectors_2d[i, 0], vectors_2d[i, 1]])
plt.title("Word2Vec Embeddings")
plt.show()'''

# TODO: Hacer esto pero con un dataset mas grande por ejemplo tweets o comentarios de peliculas
#  asi como experimentar con window y vector size

from nltk.corpus import movie_reviews
import nltk
nltk.download('movie_reviews')

# Load, tokenize, and lowercase the entire movie reviews corpus
tokenized_sentences = [[word.lower() for word in sentence] for sentence in [movie_reviews.words(fileid) for fileid in movie_reviews.fileids()]]

# Entrenar el modelo Word2Vec
model = Word2Vec(sentences=tokenized_sentences,
                 vector_size=100,
                 window=3,
                 min_count=1)

# Mostrar el vocabulario y los vectores de palabras
vocab = list(model.wv.key_to_index)
print(f"Vocab: {vocab}")

vocab = [word for word in model.wv.key_to_index if word.isalpha() and word.isascii()]
vectors = model.wv[vocab]

print(f"Vectors: {vectors}")
print()

print(f"Embedding for 'director': {model.wv['director']}")

# Reducir la dimensionalidad de los vectores de palabras
pca = PCA(n_components=3)
vectors_3d = pca.fit_transform(vectors)
# print(f"2D vectors: {vectors_2d}")

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Traemos todos los x y todos los y
ax.scatter(vectors_3d[:, 0], vectors_3d[:, 1], vectors_3d[:, 2], color='blue')

for i, word in enumerate(vocab):
    ax.text(vectors_3d[i, 0], vectors_3d[i, 1], vectors_3d[i, 2], word, usetex=False)

ax.set_title("Word2Vec Embeddings")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.show()

# TODO: investigar el tema de transformers y revisar el paper All you need is attention
# Todo en un pequeño reporte