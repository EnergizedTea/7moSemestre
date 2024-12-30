import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

class_name = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot']

'''print(x_train[0])
print(class_name[y_train[0]])
plt.imshow(x_train[0], cmap='gray')
plt.show()'''

x_train, x_test = x_train/255, x_test/255

x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                  y_train,
                                                  test_size=0.2,
                                                  random_state=42)

# TODO hacer que junto con el shape me imprima que porcentaje de x representa
print(f'{"#"*20} FINALES {"#"*20}')
print(f'Datos de Entrenamientos:    {x_train.shape}')
print(f'Datos de Validacion:    {x_val.shape}')
print(f'Datos de Prueba:    {x_test.shape}')

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28), name="Input_Layer"),
    tf.keras.layers.Dense(16, activation='relu', name="Hidden_Layer"),
    tf.keras.layers.Dense(10, activation='softmax', name="Output_Layer")
])

print(model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model_history = model.fit(x_train,
                          y_train,
                          epochs=30,
                          batch_size=64,
                          validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(loss)