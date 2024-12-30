import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, LabelEncoder

# Cargamos la base de datos
df = pd.read_csv('data.csv')


# Se nos muestran las columnas, el tipo de dato que tiene y 
# la cantidad de valores vacios en cada una
print(df.head)
print(df.info())
# Se nos muestra cuantos ceros tiene cada columna
print((df == 0).sum())
# Se nos muestran valores estadisticos de los valores de
# aquellas columnas con valores numericos
print(df.describe())

df = df.drop_duplicates()
df = df.drop('date', axis = 1)
df = df[df['price'] != 0]
df = df.drop('country', axis = 1)
df = df.drop('street', axis=1)
df['city'] = LabelEncoder().fit_transform(df['city'])
df['statezip'] = pd.to_numeric(df['statezip'].str.replace('WA ', ''))


df = df.drop('city', axis = 1)
# Binning para el sotano
bins = [0, 600, 1000, 2000, df['sqft_basement'].max()]
df['sqft_basement'] = pd.cut(df['sqft_basement'], bins = bins, labels=False)
print(df['sqft_basement'].value_counts())


# Binning para sala de estar
# bins= [500, 750, 2500, 3000, df['sqft_living'].max()]
df['sqft_living'] = pd.qcut(df['sqft_living'], q=4, labels=False)
print(df['sqft_living'].value_counts())

df['sqft_lot'] = pd.qcut(df['sqft_lot'], q=4, labels=False)
print(df['sqft_lot'].value_counts())

bins = [0, 1.5, 2, 3, df['bathrooms'].max()]
df['bathrooms'] = pd.cut(df['bathrooms'], bins=bins, labels=False)
print(df['bathrooms'].value_counts())

df = df.drop('waterfront', axis = 1)


# La matriz de correlacion nos muestra la importancia de 
# ciertos valores en el resultado de otras columnas
correlation_matrix = df.corr()
print(correlation_matrix)

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='flare', linewidths=0.5)
plt.title('Matriz de Correlación')
plt.show()

# En la variable X almacenamos las variables independientes 
# que se relacionan con el resultado
X = df.drop('price', axis=1)

# Y aqui se guarda la variable dependiente, 
# es decir, el resultado
y = df['price']

# Dividimos los datos en datos de entrenamientos y datos de 
# prueba, el primero le dara al modelo la oportunidad de 
# identificar relaciones entre las variables mientras que 
# el segundo le permitira poner a prueba las relaciones 
# encontradas

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

# Entrenamos al modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Realizamos predicciones con el modelo
y_pred = model.predict(X_test)

# Calculo de Metricas Medición
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'R2: {r2_score(y_test, y_pred)}')

# Grafica Comparativa
df_comparacion = pd.DataFrame({
    'Index': range(len(y_test)),
    'Valores Reales': y_test.values,
    'Valores Predichos': y_pred
})

plt.figure(figsize=(12, 6))
sns.lineplot(x='Index', y='Valores Reales', data=df_comparacion, label='Valores Reales', color='orange')
sns.lineplot(x='Index', y='Valores Predichos', data=df_comparacion, label='Valores Predichos', color='purple')
plt.xlabel('Índice de Muestras')
plt.ylabel('Valor')
plt.title('Comparación de y_test y y_pred')
plt.legend()
plt.grid(True)
plt.show()