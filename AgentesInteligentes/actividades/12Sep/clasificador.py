import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix

data = pd.read_csv('data.csv')

print(data.head())
print(data.describe())
print(data.info())

data = data.drop(['id', 'Unnamed: 32'], axis = 1)

label_encoder = LabelEncoder()

print(data['diagnosis'].head())
data['diagnosis'] = label_encoder.fit_transform(data['diagnosis'])
print(data['diagnosis'].head())

# Maligno es clase 1

X = data.drop('diagnosis', axis = 1)
y = data['diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Se pone en mayuscula la X para indicar que es la información de entrenamiento 
# sin alterar, cuando se manipula para mejorar rendimiento, se pasa a usar minuscula

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

sns.heatmap(conf_mat, cmap= 'Blues', annot=True, 
            xticklabels=['Benigno', 'Maligno'],
            yticklabels=['Benigno', 'Maligno'])

# plt.show()

tp = conf_mat[0][0]
fn = conf_mat[0][1]
fp = conf_mat[1][0]
tn = conf_mat[1][1]

'''
sensitivity = (tp)/(tp+fn)
print(f'Sensitivity is equal to {sensitivity}')
specifitivy = (tp)/(tn + fp)
print(f'Specifitivy is equal to {specifitivy}')
precision = (tp)/(tp+fp)
print(f'Precision is equal to {precision}')
npv = (tn)/(tn+fn)
print(f'Negative Predictive Value is equal to {npv}')
accuracy = (tp + tn)/(tn+fp+fp+fn)
'''