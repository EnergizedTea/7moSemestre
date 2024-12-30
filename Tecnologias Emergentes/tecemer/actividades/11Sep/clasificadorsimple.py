# Clasificador rudimentario de monedas
x = [
    1,0,0,0,1,
    0,1,0,0,1,
    0,0,1,0,0,
    0,1,0,1,0,
    1,0,0,0,1
]
o = [
    0,0,1,0,0,
    0,1,0,1,0,
    1,0,0,0,1,
    0,1,0,1,0,
    0,0,1,0,0
]

moneda = [
    0,0,0,0,0,
    0,1,0,1,0,
    0,0,1,0,0,
    0,1,0,1,0,
    0,0,0,0,0
]

pesos = [
    1,0,-2,0,1,
    0,1,-2,1,0,
    -2,-2,2,-2,-2,
    0,1,-2,1,0,
    1,0,-2,0,1,
]

bias = 1

def sumatoria(datos, pesos):
    n = len(datos)
    suma = 0
    for i in range(n):
        suma += datos[i] * pesos[i]
    return suma

def funcion_activacion(suma):
    return 1 if suma >= 0 else 0

def perceptron(datos, pesos, bias):
    return 'X' if funcion_activacion(sumatoria(datos, pesos) + bias) == 1 else 'O'

print(perceptron(moneda, pesos, bias))

# Para que sea automatico, los pesos deben calcularse automaticamente, 
# lo que hicimos de ajustar los pesos por prueba y error es lo que hace la maquina