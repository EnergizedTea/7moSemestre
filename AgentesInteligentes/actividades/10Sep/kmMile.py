import pandas as pd

def KmtoMile(km):
    mi = (km * 1.60934)
    return mi

def MiletoKm(mi):
    km = mi * 0.621371
    return km


db = pd.read_csv('valuesKm.csv', header = None)

data = {'Kilometros':[], 'Millas':[]}

for i in db[0]:
    print("Valor Kilometros: ", i)
    data['Kilometros'].append(float(i))
    f = KmtoMile(float(i))
    print("Valor Millas:", f)
    data['Millas'].append(float(f))


df = pd.DataFrame.from_dict(data)

df.to_csv('both.csv', index=False)