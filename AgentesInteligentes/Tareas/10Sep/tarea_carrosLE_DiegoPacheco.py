'''
Resulta con valores alejados del real pero mas cercanos que en el caso de los Dummies
MAE: 2526.4074501434384
MSE: 15916389.725439683
R2: 0.7983838478445044
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("CarPrice_Assignment.csv")
df.drop(["car_ID", "CarName"], axis=1, inplace=True)

label_encoder = {}

for columna in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[columna] = le.fit_transform(df[columna])
    label_encoder[columna] = le

X = df.drop('price', axis=1)
print(X)
y = df['price']
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