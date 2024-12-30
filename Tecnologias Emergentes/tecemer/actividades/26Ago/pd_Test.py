import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Iris.csv')

print(df.head())

# 1. Scatter Plot: Petal Length vs. Petal Width Across Species
plt.figure(figsize=(10, 6))
for species in df['Species'].unique():
    subset = df[df['Species'] == species]
    plt.scatter(subset['PetalLengthCm'], subset['PetalWidthCm'], label=species)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Length vs. Petal Width Across Species')
plt.legend()
plt.grid(True)
plt.show()

# 2. Box Plot: Distribution of Sepal Length Across Species
plt.figure(figsize=(10, 6))
df.boxplot(column='SepalLengthCm', by='Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.title('Distribution of Sepal Length Across Species')
plt.suptitle('')  # Suppress the default title
plt.grid(True)
plt.show()

# 3. Mean Sepal and Petal Dimensions by Species
mean_values = df.groupby('Species').mean()
print(mean_values[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])