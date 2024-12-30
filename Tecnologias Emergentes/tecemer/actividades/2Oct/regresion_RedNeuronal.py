# House Prices 2, Electric Boogaloo

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, normalize
import tensorflow as tf


df = pd.read_csv('data.csv')

df = df.drop('date', axis = 1)
df = df.drop_duplicates()
df = df[df['price'] != 0]
df = df.drop('country', axis = 1)
df = df.drop('waterfront', axis = 1)
df = df.drop('yr_renovated', axis=1)
df = df.drop('street', axis=1)

'''def to5(valor):
    decimal = valor * 10 % 10
    if decimal != 0 or decimal != 5:
        return round(valor * 2)/2
    
df['bathrooms'] = df['bathrooms'].apply(to5)'''

df['statezip'] = pd.to_numeric(df['statezip'].str.replace('WA ', ''))
df = df.drop('city', axis = 1)
# df = pd.get_dummies(df, columns=['city'], drop_first=True)

# Definimos las variables X e y 
X = df.drop('price', axis = 1)
y = df['price']

# Compute the correlation matrix
correlation_matrix = df.corr()
# Display the correlation of all features with 'price'
price_correlation = correlation_matrix['price'].sort_values(ascending=False)
print(price_correlation)

X = df.drop('price', axis=1)
y = df['price']

encoder = OneHotEncoder()
X = encoder.fit_transform(X).toarray()

scaler=StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_train = normalize(X_train, axis=0)
X_val = normalize(X_val, axis=0)
X_test = normalize(X_test, axis=0)

print(f'X_Train Shape:  {X_train.shape}')
print(f'X_Val Shape:    {X_val.shape}')
print(f'X_Test Shape:   {X_test.shape}')

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(40, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(80, activation='relu'),
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(1)
])

model.summary()

# Compilacion del modelo
model.compile(optimizer='adam', 
              loss = "mean_squared_error",
              metrics=['mae'])

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val))

# Evaluacion del modelo
loss, mae = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, MAE: {mae}')

# guardar el modelo
model.save('model_pach2.h5')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()