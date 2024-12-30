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

model = tf.keras.models.load_model('model_pach2.h5')
model.summary()

predictions = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(range(len(predictions)), predictions, color='blue', label='Predicted Price')
plt.plot(range(len(y_test)), y_test, color='red', alpha=0.5, label='Actual Price')
plt.title('Real vs Predictions')
plt.legend()
plt.show()