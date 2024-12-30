import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# Load the dataset
df = pd.read_csv('data.csv')

# Data cleaning
df = df.drop('date', axis=1)
df = df.drop_duplicates()
df = df[df['price'] != 0]
df = df.drop('country', axis=1)
df = df.drop('waterfront', axis=1)
df['was_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x != 0 else 0)
df = df.drop('yr_renovated', axis=1)
df = df.drop('street', axis=1)

def to5(valor):
    decimal = valor * 10 % 10
    if decimal != 0 or decimal != 5:
        return round(valor * 2) / 2

df['bathrooms'] = df['bathrooms'].apply(to5)
df['statezip'] = pd.to_numeric(df['statezip'].str.replace('WA ', ''))
df['city'] = df['city'].astype('category').cat.codes + 1

# Bins for basement
bins = [0, 500, 1000, 1500, max(df['sqft_basement'])]
df['sqft_basement'] = pd.cut(df['sqft_basement'], bins=bins, labels=False)

scalerRobust = RobustScaler()
df['sqft_living'] = scalerRobust.fit_transform(df[['sqft_living']])
df['sqft_above'] = scalerRobust.fit_transform(df[['sqft_above']])
#df['sqft_living'] = pd.cut(df['sqft_living'], bins=4, labels=False)
df['bedrooms'] = scalerRobust.fit_transform(df[['bedrooms']])
df['bathrooms'] = scalerRobust.fit_transform(df[['bathrooms']])
df['floors'] = MinMaxScaler.fit_transform(df[['floors']])


df = df.drop(['statezip', 'city', 'yr_built', 'sqft_lot', 'view'], axis=1)

# Define features and target
X = df.drop('price', axis=1)
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model with RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Hyperparameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# Predict with the best model
y_pred_scaled = best_model.predict(X_test_scaled)

# Calculate metrics
print(f'MAE: {mean_absolute_error(y_test, y_pred_scaled)}')
print(f'MSE: {mean_squared_error(y_test, y_pred_scaled)}')
print(f'R2: {r2_score(y_test, y_pred_scaled)}')

# Visualizing the correlation matrix
correlation_matrix = df.corr()
price_correlation = correlation_matrix['price'].sort_values(ascending=False)
print(price_correlation)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()
