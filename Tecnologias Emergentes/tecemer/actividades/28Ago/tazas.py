import random
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib as plt


tazas_cafe = np.array([])
lineas_codigo = np.array([])

for i in range(30):
    i +=1
    tazas_cafe = np.append(tazas_cafe, tazas_cafe[random.randint()])

for i in range(30):
    i +=1
    lineas_codigo = np.append(lineas_codigo, lineas_codigo[random.randint()])

modelo = LinearRegression()
modelo.fit(tazas_cafe.reshape(-1,1), lineas_codigo)

x = 10
tazas = np.array([x])
prediccion = modelo.predict(tazas.reshape(-1,1))

print(f'Con {x} tazas de café te salen {prediccion[0]}')

plt.scatter(tazas_cafe, lineas_codigo)
plt.xlabel('Tazas de café')
plt.ylabel('Lineas de Codigo')
plt.legend()
plt.show()