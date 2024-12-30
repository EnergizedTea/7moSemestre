import numpy as np
import pandas as pd

def km_to_miles(km):
    return np.round(km / 1.609344, 2)

def generator_dataset(n):
    km = np.random.randint(0, 280, n)
    miles = km_to_miles(km)
    data = pd.DataFrame({'km': km, 'miles': miles})
    data.to_csv('dataset_km_miles.csv', index=False)
    
if __name__ == '__main__':
    generator_dataset(100)
    print('Datos generados...')
