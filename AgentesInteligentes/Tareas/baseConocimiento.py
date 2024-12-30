padecimientos = {
    "Sistema Inmunologico" : ["Fiebre Alta"],
    "Sistema Nervioso" : ["Dolor de Cabeza", 
                          "Dolor de Cabeza Intenso",
                          "Sensibilidad a la luz",
                          "Fatiga"
                          "Visión Borrosa",
                          "Mareos"],
    "Sistema Muscular" : ["Dolor Muscular"],
    "Sistema Digestivo" : ["Aumento de la Sed", "Nauseas"],
    "Sistema Urinario" : ["Micción Frecuente"],
    "Sistema Musculoesqueletico" : ["Dolor Articular", 
                                    "Rigidez Matutina", 
                                    "Inflamación de las Articulaciones"],
}

enfermedades = {"Gripe" : 
                [padecimientos["Sistema Inmunologico"][0], padecimientos["Sistema Nervioso"][0], padecimientos["Sistema Muscular"][0]],
                "Diabetes Tipo 2" :
                [padecimientos["Sistema Digestivo"][0], padecimientos["Sistema Urinario"][0], padecimientos["Sistema Nervioso"][3]],
                "Migraña" :
                [padecimientos["Sistema Nervioso"][1], padecimientos["Sistema Digestivo"][1]],
                "Artritis Reumatoide" :
                [padecimientos["Sistema Musculoesqueletico"][0], padecimientos["Sistema Musculoesqueletico"][1], padecimientos["Sistema Musculoesqueletico"][2]],
                "Hipertension Arterial" :
                [padecimientos["Sistema Nervioso"][0], padecimientos["Sistema Nervioso"][5], padecimientos["Sistema Nervioso"][4]] 
                }

# Ahora toca motor de inferencia


