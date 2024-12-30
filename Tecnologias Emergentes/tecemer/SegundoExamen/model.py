import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

MAP_EMNIST = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',  # Digits
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T',
    30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z',  # Uppercase letters
    36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f', 41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'  # Lowercase letters
}

train = pd.read_csv('emnist-balanced-train.csv')
test = pd.read_csv('emnist-balanced-test.csv')

print(train.head())
print(test.head())

X_train = train.drop(columns=train.columns[0], axis=1)
y_train = train.iloc[:, 0]

X_test = test.drop(columns=train.columns[0], axis=1)
y_test = test.iloc[:, 0]

X_train, X_test = X_train/255, X_test/255

X_train = np.array(X_train).reshape(-1, 28, 28)
X_test = np.array(X_test).reshape(-1, 28, 28)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(47, activation='softmax')
])

# Compilar el modelo 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model_history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test)) 

# Evaluacion
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Loss: {test_loss}")
print(f"Acc: {test_acc}")

y_pred = model.predict(X_test)

plt.subplot(2,2,1)
n = random.randint(1,100)
plt.imshow(X_test[n], cmap='gray')
plt.xlabel(f"Predicción: {MAP_EMNIST[y_pred[n].argmax()]}\n Real: {MAP_EMNIST[y_test[n]]}")
plt.subplot(2,2,2)
n = random.randint(1,100)
plt.imshow(X_test[n], cmap='gray')
plt.xlabel(f"Predicción: {MAP_EMNIST[y_pred[n].argmax()]}\n Real: {MAP_EMNIST[y_test[n]]}")
plt.subplot(2,2,3)
n = random.randint(1,100)
plt.imshow(X_test[n], cmap='gray')
plt.xlabel(f"Predicción: {MAP_EMNIST[y_pred[n].argmax()]}\n Real: {MAP_EMNIST[y_test[n]]}")
plt.subplot(2,2,4)
n = random.randint(1,100)
plt.imshow(X_test[n], cmap='gray')
plt.xlabel(f"Predicción: {MAP_EMNIST[y_pred[n].argmax()]}\n Real: {MAP_EMNIST[y_test[n]]}")
plt.show()

model.save('model_emnist.h5')