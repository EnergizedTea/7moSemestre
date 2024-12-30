import pandas as pd

def CtoF(C):
    F = C * (9/5) + 32
    return F


def FtoC(F):
    C = (F - 32)(5/9)
    return C

db = pd.read_csv('random_celsius_values.csv', header = None)

data = {'Celsius':[], 'Fahrenheit':[]}

for i in db[0]:
    print("Valor celsius: ", i)
    data['Celsius'].append(float(i))
    f = CtoF(float(i))
    print("Valor farhenheit:", f)
    data['Fahrenheit'].append(float(f))

#print(data)
df = pd.DataFrame.from_dict(data)

df.to_csv('both.csv', index=False)
