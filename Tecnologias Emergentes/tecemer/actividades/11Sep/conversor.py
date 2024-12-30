import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, accuracy_score, r2_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import tensorflow as tf


df = pandas.read_csv('both.csv')

x = df.drop('Fahrenheit', axis = 1 )
y = df['Fahrenheit']

model = LinearRegression()
model.fit(x,y)


X_train, x_test, Y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state= 42)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(units = 1, input_shape = [1])
    ]
)

'''model.add(tf.keras.layers.Dense(units = 1, input_shape = [1]))
model.add(tf.keras.layers.Dense(units = 3))
model.add(tf.keras.layers.Dense(units = 1))
'''

model.compile(optimizer = tf.keras.optimizers.Adam(0.01), loss = 'mse')

#Lo de Adam es el algoritmo que toma paea poder calcular esa parte de las distancias que permite ajustar los pesos
model_history = model.fit(X_train, Y_train, epochs = 100, validation_split=0.2)

#Ahora lo que sigue es hacer la prediccion

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)

r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R2: {r2}')

plt.figure(figsize = (10, 5))
plt.plot(model_history.history['loss'], label = 'Train Loss')
plt.plot(model_history.history['val_loss'], label = 'Validation Loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluacion
celcius_input = np.array([[0]])
y_pred = model.predict(celcius_input)
print(f'Temp Celcius {celcius_input[0][0]}°C - Temp Fahrenheit {y_pred[0]}°F')