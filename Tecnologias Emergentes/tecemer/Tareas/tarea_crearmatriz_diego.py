"""
Utilizando python y ninguna otra libreria, el programa debe...

* Crear una función para generar una matriz dadas sus dimensiones
* Imprimir la matriz generada
* Crear dos matrices y multiplicarlas

"""


def generar():
    matrix = []
    rows = int(input("Insert the number of rows: "))
    columns = int(input("Insert the number of columns:  "))

    print("Insert Values")

    for row in range(rows):
        lista = []
        for column in range(columns):
            lista.append(int(input()))
        matrix.append(lista)

    return matrix


def multiplicar(m1, m2):
    if len(m1[0]) == len(m2):
        m3 = []
        for i in range(len(m1)):
            # la i aqui representa una posicion en el rango de las columnas de la primera matriz
            # Es decir, si hay cinco columnas, habran cinco i's
            rows = []
            for j in range(len(m2[0])):
                # Ahora estamos recorriendo un numero de veces que equivale a las filas de la segunda matriz
                # Ojo, ves como no escribiste en ningun momento que estas recogiendo los valores o la matriz? esto es
                # Por que gracias a len, se trata exlusivamente de asegurarnos de movernos tantas veces como hayan columnas/filas
                # Se accede al valor de cada una mas adelante

                #Iniciamos el valor valor (jsjs) en cero para que cada vez que continue el ciclo,
                #  el resultado no sea afectado por nuestras otras operaciones
                valor = 0
                for k in range(len(m2)):
                    # para realizar la multiplicacion, ahora se deben recorrer por las filas de 
                    # la segunda matriz.
                    # Con todo esto. usaremos las i,j,k que hemos recorrido para poder realizar la multiplicacion!
                    valor += m1[i][k] * m2[k][j]
                #Agegamos el valor adquirido en la fila que en estos moemntos seguimos construyendo
                rows.append(valor)
            #Una ves la fila ya tiene tantos valores como la seguna matriz tiene columnas, 
            # agregamos esta lista a la lista m3 creando una fila entera!
            m3.append(rows)

        #Una ves tenemos tantas listas como la la primera matriz tiene filas hemos terminado

        return m3
    
    else:
        print("These matrices cannot be multiplied, my apologies")
        return None


def mostrar(matrix):
    show = []
    for row in matrix:
        for i in row:
            print(i, end='|')
        print("")
    

mimatriz = generar()
mostrar(mimatriz)
m2matriz = generar()
mostrar(m2matriz)
print("----")
mostrar(mimatriz)
print("----")
matrizmulti = multiplicar(mimatriz, m2matriz)
if matrizmulti != None:
    mostrar(matrizmulti)
print("----")
