'''
Tambien resulta en varios numeros muy lejanos
MAE: 2411.093962409616
MSE: 11710105.07880737
R2: 0.8516657126363216
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("CarPrice_Assignment.csv")

df_numerico = df.select_dtypes(include=[np.number])

X = df_numerico.drop('price', axis=1)
y = df_numerico['price']

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