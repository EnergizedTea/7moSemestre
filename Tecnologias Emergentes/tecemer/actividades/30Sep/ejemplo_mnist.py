import tensorflow as tf
import matplotlib.pyplot as plt
import random

mnist = tf.keras.datasets.mnist

# print(mnist)
n = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalmente limpieza pero ya esta limpio

'''print(x_train[n])
plt.imshow(x_train[n], cmap='gray')
plt.show()'''

# Hay que manipular los valores para normalizarlos

x_train, x_test = x_train/255, x_test/255

model = tf.keras.models.Sequential([
    # Capa de entrada
    tf.keras.layers.Flatten(input_shape=(28,28)),
    # Capa oculta
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),

    # Capa de Salida
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilar el modelo 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entrenar el modelo
model_history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test)) 

# Evaluacion
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Loss: {test_loss}")
print(f"Acc: {test_acc}")

'''
plt.figure(figsize=(15,4))
plt.plot(1,2,1)
plt.plot(model_history.history['accuracy'], label='Accuracy')
plt.plot(model_history.history['valaccuracy'], label='Val Acc')
plt.xlabel('Epochs')
plt.ylabel('Acc')

plt.figure(figsize=(15,4))
plt.plot(model_history.history['loss'], label='Loss')
plt.plot(model_history.history['val_loss'], label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
'''
y_pred = model.predict(x_test)

plt.subplot(2,2,1)
n = random.randint(1,100)
plt.imshow(x_test[n], cmap='gray')
plt.xlabel(f"Predicción: {y_pred[n].argmax()}")
plt.subplot(2,2,2)
n = random.randint(1,100)
plt.imshow(x_test[n], cmap='gray')
plt.xlabel(f"Predicción: {y_pred[n].argmax()}")
plt.subplot(2,2,3)
n = random.randint(1,100)
plt.imshow(x_test[n], cmap='gray')
plt.xlabel(f"Predicción: {y_pred[n].argmax()}")
plt.subplot(2,2,4)
n = random.randint(1,100)
plt.imshow(x_test[n], cmap='gray')
plt.xlabel(f"Predicción: {y_pred[n].argmax()}")
plt.show()

model.save('model_mnist.h5')