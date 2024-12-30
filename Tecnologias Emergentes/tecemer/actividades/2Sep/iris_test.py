#Holi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

iris = pd.read_csv("Iris.csv")
# Quitamos la primera columna por que son solo indices
iris = iris.drop("Id", axis = 1)

'''
print(iris.describe())

fig = iris[iris.Species == 'Iris-setosa'].plot(kind="scatter",
                                               x="SepalLengthCm",
                                               y="SepalWidthCm", 
                                               color='red')

iris[iris.Species == 'Iris-versicolor'].plot(kind="scatter",
                                               x="SepalLengthCm",
                                               y="SepalWidthCm", 
                                               color='blue',
                                               ax=fig)

iris[iris.Species == 'Iris-virginica'].plot(kind="scatter",
                                               x="SepalLengthCm",
                                               y="SepalWidthCm", 
                                               color='yellow',
                                               ax=fig)

plt.show()
'''
'''
sns.pairplot(iris, hue='Species')
#plt.savefig('dispersion.png')
plt.show()
'''

# Estos primeros algoritmos de machine learning son los mas 
# simples y se utilizan para información senilla

# Utilizaremos "Vecino más cercano"

X = iris.drop('Species', axis=1)
y = iris['Species']

#Ya estan separados los datos entre su especie y sus datos

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size = 0.2, 
                                                    random_state=12)

# Construccion del Modelo KNN
knn = KNeighborsClassifier(n_neighbors = 3)

# Entrenar el modelo
knn.fit(X_train, y_train)

#  Predicción 
y_pred = knn.predict(X_test)

# Evaluacion

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Salio 1... es un overfitting


# Ahora intentemos con regresión logistica

# Construccion del modelo: Regresión Lógistica
model_lg = LogisticRegression()
model_lg.fit(X_train, y_train)

# Volvemos a realizar predicciones
y_pred_lg = model_lg.predict(X_test)
print(f'Accuracy LogisticRegression: {accuracy_score(y_test, y_pred_lg)}')

# Construcción del Modelo: Arbol de Clasificacion de Decisiones
model_dt = DecisionTreeClassifier()
model_dt.fit(X_train, y_train)
y_pred_dt = model_dt.predict(X_test)
print(f'Accuracy DecisionTreeClassifier: {accuracy_score(y_test, y_pred_dt)}')

# Construccion del Modelo: SVC

model_svc = SVC()
model_svc.fit(X_train, y_train)
y_pred_svc = model_svc.predict(X_test)
print(f'Accuracy LogisticRegression: {accuracy_score(y_test, y_pred_svc)}')

# Mmm... nos sigue dando uno, la base de datos es 
# buena pero tampoco tan buena como para darnos un 100, 
# Al cambiar el seed de 42 a 12 funciona, 
# pero eso no indica que solo sea la semilla 

# Hay que analizar que es lo que nos falta 