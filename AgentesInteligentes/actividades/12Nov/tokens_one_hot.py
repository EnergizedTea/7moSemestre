from sklearn.preprocessing import OneHotEncoder
import numpy as np

frase = "Los alumnos estan preocupados por los proyectos finales"

tokens = frase.split()
print(tokens)

tokens_array = np.array(tokens).reshape(-1, 1)
print(tokens_array)

encoder = OneHotEncoder(sparse_output=False)
one_hot = encoder.fit_transform(tokens_array)

print(f"Palabras: {one_hot.ravel()}")
print(one_hot)
