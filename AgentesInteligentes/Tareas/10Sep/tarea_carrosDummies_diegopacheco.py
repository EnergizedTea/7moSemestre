# Resultados: Valores muuuuy lejos del real

'''MAE: 4441.9429888842205
MSE: 46278034.08509429
R2: 0.4137867115276461'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("CarPrice_Assignment.csv")
df['symboling'] = df['symboling'].astype('object')

df.drop("car_ID", axis = 1, inplace=True)
df_procesado = pd.get_dummies(df, drop_first=True)

X = df_procesado.drop('price', axis=1)
print(X)
y = df_procesado['price']
print(y)

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)

# Crear el modelo

model = LinearRegression()

# Entrenar el modelo
model.fit(X_train, y_train)

# Predicción
y_pred = model.predict(X_test)

# Metricas
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'R2: {r2_score(y_test, y_pred)}')

print(f'Reales: {y_test.values}')
print(f'Pred: {y_pred}')