import tensorflow as tf
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

nltk.download('punkt')

sentences = (
    ("Enjoyed every moment of my friends and family.", "positive"),
    ("Couldn’t be more satisfied with my recent experience.", "positive"),
    ("The worst decision was the terrible quality.", "negative"),
    ("Extremely disappointed in the food here.", "negative"),
    ("Feeling really let down by the lack of support.", "negative"),
    ("I can't stand the terrible quality.", "negative"),
    ("Enjoyed every moment of my friends and family.", "positive"),
    ("The worst decision was the terrible quality.", "negative"),
    ("I absolutely love the service here.", "positive"),
    ("Extremely disappointed in the buggy app.", "negative"),
    ("Couldn’t be more satisfied with the cozy atmosphere.", "positive"),
    ("This is beyond disappointing, the entire experience.", "negative"),
    ("Can't believe how awful the food here.", "negative"),
    ("Highly recommend the cozy atmosphere.", "positive"),
    ("Extremely disappointed in the food here.", "negative"),
    ("Enjoyed every moment of the support I received.", "positive"),
    ("The best experience with the cozy atmosphere.", "positive"),
    ("Feeling really let down by the lack of support.", "negative"),
    ("Highly recommend how well things turned out!", "positive"),
    ("Couldn’t be more satisfied with my recent experience.", "positive"),
)

x = [word_tokenize(sentence.lower()) for sentence, _ in sentences]
y = np.array([1 if label == 'positive' else 0 for _, label in sentences])

print(f'x: {x}')    
print(f'y: {y}')    

# Entrenar el modelo Word2Vec
model_w2v = Word2Vec(sentences=x,
                     vector_size=10,
                     window=3,
                     min_count=1)

def sentence_to_vector(sentence):
    return np.mean([model_w2v.wv[word] for word in sentence if word in model_w2v.wv], axis=0)   

x_vectors = np.array([sentence_to_vector(sentence) for sentence in x])
print(f'x_vectors: {x_vectors}')

x_train, x_test, y_train, y_test = train_test_split(x_vectors, 
                                                    y, 
                                                    test_size=0.2,
                                                    random_state=42)

# creamos nuestro bellisimo modelo
model = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(1, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')]
)

print(model.summary())

model.compile(optimizer='adam',
              loss = 'binary_crossentropy', # por que solo hay dos salidas
              metrics = ['accuracy'])

history_model = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)
print(f"Loss: {loss}, Acc: {acc}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history_model.history['accuracy'], label='Train ACC')
plt.plot(history_model.history['val_accuracy'], label='Val ACC')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history_model.history['loss'], label='Train Loss')
plt.plot(history_model.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

new_sentece = "I was absolutely thrilled with the outstanding customer service"
new_sentece_tokens = word_tokenize(new_sentece.lower())
new_sentece_vector = sentence_to_vector(new_sentece_tokens)
new_sentece_vector = np.array([new_sentece_vector])

prediction = model.predict(new_sentece_vector)
print(f"Prediction: {prediction}")
print("Positive" if prediction > 0.5 else "Negative")

# Tarea: Clasificador base con un dataset mucho mas robusto
