import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('chitai_gorod_dataset.csv', sep=';', encoding='utf-16')

# print(df.head())
# print(df.describe())
print(df.columns)

# print(df[df['height'] == df.height.max()].index.values)
# print(df[df['weight'] == df.weight.max()].index.values)

df.drop([df[df['height'] == df.height.max()].index.values[0], df[df['weight'] == df.weight.max()].index.values[0]], inplace=True)

# sns.set(style="ticks", color_codes=True)
sns.pairplot(df[['height', 'length', 'weight']], palette="husl")

height = df[(df['height'] < df['height'].quantile(.9)) & (df['height'] > df['height'].quantile(.1))]['height'].dropna()

hist, bins = np.histogram(height, bins=2)
# print('Hist = ', hist)
# print('Proportion = ', hist[0] / hist[1])
# print('bins = ', bins)

print('Height of the main shelves (plus 3 cm.) = ', round((bins[1] + 3), 1))
print('Height of additional (big) shelves (plus 3 cm.) = ', round((height.max() + 3), 1))
print(height[height > bins[1]].count())
print(height[height < bins[1]].count())

prices = df[(df['price'] < df['price'].quantile(.9)) & (df['price'] > df['price'].quantile(.1))]['price'].dropna()
hist, bins = np.histogram(prices)
print('')
print(hist)
print(np.where(hist == hist.max()))
print(bins)
print('Most common price = ', np.average([bins[np.where(hist == hist.max())[0]], bins[np.where(hist == hist.max())[0] + 1]]))
print('Median price = ', np.median(prices.array))
print('Average price = ', np.mean(prices.array))

print('\nTop authors by books number:')
print(df.groupby(by='author').count()['title'].sort_values(ascending=False).head())
print('\nTop books by release year:')
print(df.groupby(by='year').count()['title'].sort_values(ascending=False).head())
print('\nTop books by page number:')
pages = df.groupby(by='pages').count()['title'].sort_values(ascending=False)
print('\n', pages.head())
page_conspiracy = df[df['pages'] == pages.index[0]].groupby(by='publisher').count()['title'].sort_values(ascending=False)
print('\n', page_conspiracy.head())
print('\nMost lazy publisher is: ', page_conspiracy.index[0])

print('\nBooks ratings:')
print(df.groupby(by='book_pg').count()['title'].sort_values(ascending=False).head()) ############ ONLY FOR CHITAI GOROD

# sns.distplot(prices.array, bins=10)
# sns.distplot(height, bins=2)
# sns.jointplot(x='height', y='length', data=df)

plt.show()
