import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random

model = tf.keras.models.load_model("model_mnist.h5")
test = Image.open('test.png').convert('L')
test = test.resize((28, 28))
test = np.array(test) 
test = test / 255.0
test = test.reshape(1, 28, 28, 1)

y_pred = model.predict(test)

plt.subplot(1,1,1)
n = random.randint(1,100)
plt.imshow(test.reshape(28,28), cmap='gray')
plt.xlabel(f"Predicción: {y_pred.argmax()}")
plt.show()