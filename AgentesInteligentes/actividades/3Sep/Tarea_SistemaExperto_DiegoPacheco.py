sintomas = {
    "Sistema Inmunologico" : ["Fiebre Alta"],
    "Sistema Nervioso" : ["Dolor de Cabeza", 
                          "Dolor de Cabeza Intenso",
                          "Sensibilidad a la luz",
                          "Fatiga",
                          "Visión Borrosa",
                          "Mareos"],
    "Sistema Muscular" : ["Dolor Muscular"],
    "Sistema Digestivo" : ["Aumento de la Sed", "Nauseas"],
    "Sistema Urinario" : ["Micción Frecuente"],
    "Sistema Musculoesqueletico" : ["Dolor Articular", 
                                    "Rigidez Matutina", 
                                    "Inflamación de las Articulaciones"],
}

enfermedades = {
    "Gripe": ["Fiebre Alta", "Dolor de Cabeza", "Dolor Muscular"],
    "Diabetes Tipo 2": ["Aumento de la Sed", "Micción Frecuente", "Fatiga"],
    "Migraña": ["Dolor de Cabeza Intenso", "Nauseas"],
    "Artritis Reumatoide": ["Dolor Articular", "Rigidez Matutina", "Inflamación de las Articulaciones"],
    "Hipertension Arterial": ["Dolor de Cabeza", "Mareos", "Visión Borrosa"]

}

# Ahora toca motor de inferencia

# Sintomas corroborados de existir en la base de conocimiento
corSintomas = []

# Enfermedades que podria tener el paciente, su llave siendo el nombre
enferPac = {}

print("--------- Agregar Sintomas -----------")
print("Ingrese Sintomas, o Ingrese 'Listo' para terminar de agregar Sintomas\n")
opc = input

while opc != "Listo": 
    for sistema in sintomas:
        for sintoma in sintomas[sistema]:
            if opc == sintoma:
                corSintomas.append(opc)

    opc = input()

#print(corSintomas)
        
for sintoma in corSintomas:
    for enfermedad in enfermedades:
        if sintoma in enfermedades[enfermedad]:
            if enfermedad in enferPac:
                enferPac[enfermedad] += 1
            else:
                enferPac[enfermedad] = 1

#print(enferPac)
        
print("Enfermedades en base junto con el numero de sintomas convergentes:   \n")
match = max(enferPac, key=enferPac.get)
for enfermedad, conteo in enferPac.items():
    if enfermedad == match:
        print(f"La enfermedad {enfermedad} contiene {conteo} de los sintomas que usted a indicado, lo cual la convierte en la enfermedad mas posible.")
    else:
        print(f"La enfermedad {enfermedad} contiene {conteo} de los sintomas que usted a indicado.")

