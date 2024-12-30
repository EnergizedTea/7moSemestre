# Ahora programamos para aprendizaje de maquina
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_squared_error, r2_score

data = pd.read_csv('both.csv')

X = data.drop('Fahrenheit', axis = 1)
y = data['Fahrenheit']

model = LinearRegression()
model.fit(X, y)

celcius_input = np.array([[0]])

y_pred = model.predict(celcius_input)

print(f'Temp Celcius {celcius_input[0][0]}°C - Temp Fahrenheit {y_pred[0]}°F')


# ahora vamos a entrenarlo como habiamos hecho anteriormente

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state=12)

model = LinearRegression()
model.fit(X_train, y_train)

# Prediccion
y_pred = model.predict(X_test)

# Evaluacion
celcius_input = np.array([[0]])
y_pred = model.predict(celcius_input)
print(f'Temp Celcius {celcius_input[0][0]}°C - Temp Fahrenheit {round(y_pred[0])}°F')

