'''
Seleccion y Analisis Machine Learning

Debido a la naturaleza del problema, es ideal resolver esta problematica con
un Modelo de Aprendizaje Supervisado de tipo de Regresión, esto se debe al hecho de buscar
predecir un solo valor numerico basandonos en los patrones existentes en los ejemplos de
la base de datos

Aunque esto puede ser resuelto con redes neuronales, se optara primero por la regresión lineal
debido a que, aunque se trata de una alta cantidad de datos, no son suficientes como para 
justificar el coste de procesar redes neuronales si regresion lineal resulta ser suficiente.
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
df = pd.read_csv('data.csv')

'''
Iniciamos con la limpieza de la base de datos, primero eliminamos la columna date, 
pues esta solo hace referencia a la fecha en que se agrego el dato a la base.

En caso de existir datos duplicados, los eliminamos.

Despues, continuamos eliminando todo valor en precio que no sea mayor a 0.0

Tambien eliminaremos la columna pais debido a que se trata siempre del mismo

Debido a la existencia de muy pocas propiedades con vista a un cuerpo acuatico, se decide eliminar
esta columna en el proceso de limpieza


Un analisis de la columna de año en que se renova nos permite ver que se nos da ya sea un año o 0, 
para evitar confusiones se simplificara a una columna de resultado binario en que se pregunta si se 
ha restaurado o no. Esto es permisible gracias a que la columna 'condition' nos sirve como mejor metrica
del estado general de la casa.

Las calles debido a tener varios factores dificiles de procesar son eliminados, ya que si el modelo se le
da un inmueble con una calle nueva, tendra que trabajar con los demas valores existentes. La existencia del
codigo postal y la ciudad permitiran complementar al modelo de mejor manera

Analizando la columna respectiva a los baños notamos la existencia de "Baños 0.25, 0.75 y 1.75.", por ello
se hara limpieza de estos. asumimos como baño de 0.25 tal vez un baño que solo tiene un retrete o solo un lavabo
para evitar confundir al modelo, seran remplazados por el digito 0.5. Para esto se crea una funcion que realice 
esto por su cuenta

Los codigos zip son todos de washington, debido a esto se dejara el numero zip sin indicar el estado

Por ultimo, las ciudades son de tipo string, para poder procesarlos las convertiremos a numeros por medio de
label encoding

Con esto terminamos la limpieza del codigo y podemos empezar a entrenar el modelo
'''

df = df.drop('date', axis = 1)
df = df.drop_duplicates()
df = df[df['price'] != 0]
df = df.drop('country', axis = 1)
df = df.drop('waterfront', axis = 1)
df['was_renovated'] = df['yr_renovated'].apply(lambda x:1 if x != 0 else 0)
df = df.drop('yr_renovated', axis=1)
df = df.drop('street', axis=1)

def to5(valor):
    decimal = valor * 10 % 10
    if decimal != 0 or decimal != 5:
        return round(valor * 2)/2
    
df['bathrooms'] = df['bathrooms'].apply(to5)
df['statezip'] = pd.to_numeric(df['statezip'].str.replace('WA ', ''))
df['city'] = df['city'].astype('category').cat.codes + 1

# Definimos las variables X e y 
X = df.drop('price', axis = 1)
y = df['price']

# Separacion de datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state=42)

model = LinearRegression()
# Entrenamiento del modelo
model.fit(X_train, y_train)
# Prediccion
y_pred = model.predict(X_test)

# Metricas
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'R2: {r2_score(y_test, y_pred)}')

# Compute the correlation matrix
correlation_matrix = df.corr()
# Display the correlation of all features with 'price'
price_correlation = correlation_matrix['price'].sort_values(ascending=False)
print(price_correlation)

'''
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()
'''

# La matriz de correlacion nos muestra combinaciones de valores junto con el 
# impacto que estos tienen en el modelo, esto nos permite ver que valores tienen 
# mayor y menor impacto

# Una sorpresa fue el descubrir que el modelo interpreta el valor ciudad como 
# un valor de baja correlacion, viendo la información almacenada dentro de df
# Hace que me pregunte si esto se debe al hecho de estar almacenados como un
# numero. 
# Para poner a prueba esta teoria se ha optado por cambiar a one-hot encoding
# brevemente

"""
city_Mercer Island          0.149117
city_Bellevue               0.138463
city_Medina                 0.129952
city_Clyde Hill             0.083508
sqft_lot                    0.051347
city_Sammamish              0.050838
city_Redmond                0.045415
condition                   0.038892
city_Kirkland               0.034390
city_Seattle                0.033815
city_Yarrow Point           0.033503
yr_built                    0.021757
city_Newcastle              0.016391
city_Woodinville            0.016205
city_Issaquah               0.015179
city_Fall City              0.011765
city_Beaux Arts Village     0.004919
city_Preston                0.000169
city_Snoqualmie Pass       -0.000865
city_Snoqualmie            -0.001332
city_Normandy Park         -0.002313
city_Ravensdale            -0.003051
city_Inglewood-Finn Hill   -0.003494
city_Carnation             -0.006076
city_Milton                -0.010148
city_Bothell               -0.011589
city_Vashon                -0.012121
city_Lake Forest Park      -0.012778
city_Black Diamond         -0.013087
city_Skykomish             -0.014799
city_Pacific               -0.021436
city_Kenmore               -0.022101
city_Duvall                -0.026344
was_renovated              -0.028839
city_North Bend            -0.029597
city_Enumclaw              -0.032731
city_Tukwila               -0.035451
city_Shoreline             -0.040646
city_Kent                  -0.042237
city_SeaTac                -0.042399
city_Covington             -0.043582
city_Burien                -0.044805
city_Des Moines            -0.050961
city_Maple Valley          -0.056422
city_Renton                -0.082633
city_Federal Way           -0.084298
city_Auburn                -0.091094
"""

# Despues de realizar esto, ae descubre lo siguiente:
# Aunque uno podria asumir que la ciudad en la que una casa se encuentra 
# tendra un gran impacto en el valor final de la misma, en el caso de nuestra 
# base de datos es un impacto minimo e innecesario, para poder permitirle al
# modelo enfocar sus esfuerzos en otros valores, este y otras columnas seran 
# eliminadas

dfEmpty = df.eq(0).sum()
print(dfEmpty)
print(df.shape[0])

# Un analisis mucho mas extenso de los valores en la base de datos 
# nos muestra que de los 4551 valores...
# * 2718 casas no tienen un sotano
# * 2706 casas no han sido renovadas
# * 4103 tienen una vista de 0
# Esto junto con la matriz de relacion nos ayuda para poder realizar mas limpieza de datos

# Tiene una relacion con el precio de -0.029
df = df.drop('was_renovated', axis = 1)

# Tiene una relacion con el precio de -0.047
df = df.drop('statezip', axis = 1)

# Tiene una relacion con el precio de 0.016
df = df.drop('city', axis = 1)

# Tiene una relacion con el precio de 0.022
df = df.drop('yr_built', axis = 1)

# Tiene una relacion con el precio de 0.039
df = df.drop('condition', axis = 1)

# Tiene una relacion con el precio de 0.051
df = df.drop('sqft_lot', axis = 1)

# Definimos las variables X e y 
X = df.drop('price', axis = 1)
y = df['price']

'''
Ahora que hemos realizado la limpieza podemos entrenar 
el modelo y anotar cualquier cambio notado
'''

# Separacion de datos para entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

model = LinearRegression()

# Entrenamiento del modelo
model.fit(X_train, y_train)

# Prediccion
y_pred = model.predict(X_test)

# Metricas
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'R2: {r2_score(y_test, y_pred)}')

'''# Compute the correlation matrix
correlation_matrix = df.corr()
# Display the correlation of all features with 'price'
price_correlation = correlation_matrix['price'].sort_values(ascending=False)
print(price_correlation)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()'''

'''
MAE: 167475.87148773397
MSE: 65850129746.76615
R2: 0.5573707278589556

R2 bajo... a continuacion intentaremos realizar normalización
'''

scaler = StandardScaler()

# Ajustar y transformar los datos de entrenamiento
X_train_scaled = scaler.fit_transform(X_train)

# Solo transformar los datos de prueba
X_test_scaled = scaler.transform(X_test)

# Entrenar el modelo con los datos normalizados
model = RandomForestRegressor(n_estimators=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Predicción con los datos normalizados
y_pred_scaled = model.predict(X_test_scaled)

# Métricas
print(f'MAE: {mean_absolute_error(y_test, y_pred_scaled)}')
print(f'MSE: {mean_squared_error(y_test, y_pred_scaled)}')
print(f'R2: {r2_score(y_test, y_pred_scaled)}')

# Compute the correlation matrix
correlation_matrix = df.corr()
# Display the correlation of all features with 'price'
price_correlation = correlation_matrix['price'].sort_values(ascending=False)
print(price_correlation)
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.show()

#  Debido al hecho de no notar ningun cambio real, pasaremos a trabajar con 
# un modelo distinto, Random Forest Regression

#Random forest regression