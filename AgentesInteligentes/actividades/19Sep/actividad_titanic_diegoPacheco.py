# Analisis del dataset de titanic
# El analisis teorico fue realizado junto con David Bojalil Abiti
# mas sin embargo, la actividad practica se realizo de forma individual


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Primero, limpieza,

# Un analisis previo de las columnas existentes nos llevo
# a la decision de considerar esta categorias como las mas
# Importantes: sex, pclass, age, parch, cabin y supervivencia

dtest = pd.read_csv("test.csv")
dtrain = pd.read_csv("train.csv")


print(dtest.describe())
print(dtest.head())

'''
   PassengerId  Pclass                                          Name     Sex   Age  SibSp  Parch   Ticket     Fare Cabin Embarked
0          892       3                              Kelly, Mr. James    male  34.5      0      0   330911   7.8292   NaN        Q
1          893       3              Wilkes, Mrs. James (Ellen Needs)  female  47.0      1      0   363272   7.0000   NaN        S
2          894       2                     Myles, Mr. Thomas Francis    male  62.0      0      0   240276   9.6875   NaN        Q
3          895       3                              Wirz, Mr. Albert    male  27.0      0      0   315154   8.6625   NaN        S
4          896       3  Hirvonen, Mrs. Alexander (Helga E Lindqvist)  female  22.0      1      1  3101298  12.2875   NaN        Ss

       PassengerId      Pclass         Age       SibSp       Parch        Fare
count   418.000000  418.000000  332.000000  418.000000  418.000000  417.000000
mean   1100.500000    2.265550   30.272590    0.447368    0.392344   35.627188
std     120.810458    0.841838   14.181209    0.896760    0.981429   55.907576
min     892.000000    1.000000    0.170000    0.000000    0.000000    0.000000
25%     996.250000    1.000000   21.000000    0.000000    0.000000    7.895800
50%    1100.500000    3.000000   27.000000    0.000000    0.000000   14.454200
75%    1204.750000    3.000000   39.000000    1.000000    0.000000   31.500000
max    1309.000000    3.000000   76.000000    8.000000    9.000000  512.329200
'''

na_Test = dtest.isnull().sum()
print(na_Test)
na_Train = dtrain.isnull().sum()
print(na_Train)

'''
PassengerId      0
Pclass           0
Name             0
Sex              0
Age             86
SibSp            0
Parch            0
Ticket           0
Fare             1
Cabin          327
# de los 418 datos de prueba, 327 no se conoce la cabina
Embarked         0
dtype: int64
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
# de los 891 datos de entrenamiento, 687 no se conoce la cabina
Embarked         2
dtype: int64

Debido a la ausencia tan grande de datos de cabina, directamente 
eliminamos la columna cabina
'''

label_encoder = LabelEncoder()

dTestCleaned = dtest.drop(['Cabin', 'PassengerId','SibSp', 'Ticket', 'Fare', 'Embarked'], axis = 1).dropna()
dTestCleaned['Sex'] = label_encoder.fit_transform(dTestCleaned['Sex'])

dTrainCleaned = dtrain.drop(['Cabin', 'PassengerId', 'Name', 'SibSp', 'Ticket', 'Fare', 'Embarked'], axis = 1).dropna()
dTrainCleaned['Sex'] = label_encoder.fit_transform(dTrainCleaned['Sex'])

'''na_Train = dTrainCleaned.isnull().sum()
print(na_Train)
na_Test = dTestCleaned.isnull().sum()
print(na_Test)'''

print(dTrainCleaned.head())

Xt = dTrainCleaned.drop('Survived', axis=1)
yt = dTrainCleaned['Survived']

X_train, X_test, y_train, y_test = train_test_split(Xt,
                                                    yt,
                                                    test_size=0.2,
                                                    random_state=42)

'''scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Metricas
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'MSE: {mean_squared_error(y_test, y_pred)}')
print(f'R2: {r2_score(y_test, y_pred)}')

plt.figure(figsize=(10, 6))
plt.plot(range(len(y_pred)), y_pred, color='blue', label='Supervivencia Predicha')
plt.plot(range(len(y_test)), y_test, color='red', alpha=0.5, label='Supervivencia Real')
plt.title('Supervivencia Predicha vs Supervivencia Real')
plt.legend()
plt.show()

print(f'Reales: {y_test.values}')
print(f'Pred: {y_pred}')'''

# Construccion del modelo: Regresión Lógistica
model = LogisticRegression()
model.fit(X_train, y_train)

# Volvemos a realizar predicciones
y_pred = model.predict(X_test)
print(f'Accuracy LogisticRegression: {accuracy_score(y_test, y_pred)}')

# Crear gráfica de comparación
plt.figure(figsize=(10, 6))
plt.plot(range(len(y_pred)), y_pred, color='blue', label='Supervivencia Predicha (Logistic Regression)')
plt.plot(range(len(y_test)), y_test, color='navy', alpha=0.5, label='Supervivencia Real')
plt.title('Comparación: Supervivencia Predicha vs Supervivencia Real')
plt.legend()
plt.show()

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Sobrevivió', 'Sobrevivió'], yticklabels=['No Sobrevivió', 'Sobrevivió'])
plt.ylabel('Valores Reales')
plt.xlabel('Predicciones')
plt.title('Matriz de Confusión')
plt.show()

y_pred = model.predict(dTestCleaned.drop('Name', axis=1))

for i in range(len(y_pred)):
    nombre = dTestCleaned['Name'].iloc[i]
    estado = "sobrevivio" if y_pred[i] == 1 else 'no sobrevivio'
    print('\n')
    print(f"{nombre}, de acuerdo a nuestro modelo, {estado}.")

