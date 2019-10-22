import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('chitai_gorod_dataset.csv', sep=';', encoding='utf-16')

# print(df.head())
# print(df.describe())
print(df.columns)

print(df[df['height'] == df.height.max()].index.values)
print(df[df['weight'] == df.weight.max()].index.values)

df.drop([df[df['height'] == df.height.max()].index.values[0], df[df['weight'] == df.weight.max()].index.values[0]], inplace=True)

# sns.set(style="ticks", color_codes=True)
# sns.pairplot(df[['height', 'length', 'width', 'weight']], palette="husl")

height = df[(df['height'] < df['height'].quantile(.9)) & (df['height'] > df['height'].quantile(.1))]['height'].dropna()

hist, bins = np.histogram(height, bins=2)
print('Hist = ', hist)
print('Proportion = ', hist[0] / hist[1])
print('bins = ', bins)

print('Height of the main shelves (plus 3 cm.) = ', round((bins[1] + 3), 1))
print('Height of additional (big) shelves (plus 3 cm.) = ', round((height.max() + 3), 1))
print(height[height > bins[1]].count())
print(height[height < bins[1]].count())

sns.distplot(height, bins=2)
# sns.jointplot(x='height', y='length', data=df)

plt.show()
