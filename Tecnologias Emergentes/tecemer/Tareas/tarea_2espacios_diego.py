# Diego Pacheco Valdez
# Tengo entendido que la idea de la tarea no es 
# tanto que la maquina se ponga a ver la logica

islaInicio = []
islaFinal = []

def funcion(i1 = [], i2 = []):
    print('---------------')
    print("Isla del Inicio: ",i1)
    print("Isla del final:  ", i2)
    while len(i2) != 3:
        try:
            i1.remove('P')
            i2.append("P")
            print('---------------')
            print("Isla del Inicio: ",i1)
            print("Isla del final:  ", i2)
        except:
            try:
                print('---------------')
                i1.remove('A')
                i2.append('A')
                print("Isla del Inicio: ",i1)
                print("Isla del final:  ", i2)
            except:
                    i2.remove('P')
                    i1.append('P')
                    print("Isla del Inicio: ",i1)
                    print("Isla del final:  ", i2)
                    print('---------------')
                    i1.remove('Z')
                    i2.append('Z')
                    print("Isla del Inicio: ",i1)
                    print("Isla del final:  ", i2)

while len(islaInicio) != 3:
    islaInicio.append(input("Ingrese valor:   "))
funcion(islaInicio, islaFinal)
                