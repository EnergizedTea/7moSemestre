import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random

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
    # Capa de entrada
    tf.keras.layers.Flatten(input_shape=(28,28)),
    # Capa oculta
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),

    # Capa de Salida
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
plt.xlabel(f"Predicción: {y_pred[n].argmax()}")
plt.subplot(2,2,2)
n = random.randint(1,100)
plt.imshow(X_test[n], cmap='gray')
plt.xlabel(f"Predicción: {y_pred[n].argmax()}")
plt.subplot(2,2,3)
n = random.randint(1,100)
plt.imshow(X_test[n], cmap='gray')
plt.xlabel(f"Predicción: {y_pred[n].argmax()}")
plt.subplot(2,2,4)
n = random.randint(1,100)
plt.imshow(X_test[n], cmap='gray')
plt.xlabel(f"Predicción: {y_pred[n].argmax()}")
plt.show()

model.save('model_emnist.h5')