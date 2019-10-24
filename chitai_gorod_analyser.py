'''
Датасет был любезно предоставлен (нет) сайтом магазина читай-город.
Этот анализ затевался изначально просто чтобы посчитать среднюю высоту книги для строительства (очередного) шкафчика.
Но раз парсить, то парсить всё что можно. Поэтому помимо габаритов книги были собраны еще кое-какие данные.
Парсера тут нет, датасет уже собран.
'''

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

df = pd.read_csv('chitai_gorod_dataset.csv', sep=';', encoding='utf-16')

print('Количество записей: ', df.shape[0])

''' Обрезаем аномалии по высоте книги и весу, которые попадают за 99ый и перед 1ый квантилем.
Попадаются всякие адские фолианты или гигантские календари (которые почему-то у читай-города в разделе "книги")
'''
df = df[(df['height'] < df['height'].quantile(.999)) & (df['height'] > df['height'].quantile(.001))]
df = df[(df['weight'] < df['weight'].quantile(.999)) & (df['weight'] > df['weight'].quantile(.001))]

sns.pairplot(df[['height', 'length', 'weight']], kind='reg')
plt.show()

height = df['height'].dropna()
hist, bins = np.histogram(height, bins=2)
# print('Hist = ', hist)
# print('Proportion = ', hist[0] / hist[1])
# print('bins = ', bins)

print('Height of the main shelves (plus 3 cm.) = ', round((bins[1] + 3), 1))
print('Height of additional (big) shelves (plus 3 cm.) = ', round((height.max() + 3), 1))
print(height[height > bins[1]].count())
print(height[height < bins[1]].count())

sns.distplot(height, bins=2)
plt.show()

# prices = df[(df['price'] < df['price'].quantile(.9)) & (df['price'] > df['price'].quantile(.1))]['price'].dropna()
prices = df['price'].dropna()
hist, bins = np.histogram(prices)
print('')
print('Most common price = ', np.average([bins[np.where(hist == hist.max())[0]], bins[np.where(hist == hist.max())[0] + 1]]))
print('Median price = ', np.median(prices.array))
print('Average price = ', np.mean(prices.array))

print('\nTop authors by books number:')
print(df.groupby(by='author').count()['title'].sort_values(ascending=False).head())
print('\nTop books by release year:')
print(df.groupby(by='year').count()['title'].sort_values(ascending=False).head())
print('\nTop books by page number:')
pages = df.groupby(by='pages').count()['title'].sort_values(ascending=False)
print(pages.head())
page_conspiracy = df[df['pages'] == pages.index[0]].groupby(by='publisher').count()['title'].sort_values(ascending=False)
print('\n', page_conspiracy.head())
print('\nMost lazy publisher is: ', page_conspiracy.index[0])

############ ONLY FOR CHITAI GOROD ############
print('\nBooks ratings:')
print(df.groupby(by='book_pg').count()['title'].sort_values(ascending=False).head()) 

print('\nСамая дорогая книга:')
print(df[df['price'] == df['price'].max()][['author', 'title', 'price']])
sns.regplot(x='weight', y='price', data=df)
plt.show()

# sns.distplot(prices.array, bins=10)
# sns.boxplot(prices.array, hue=0.7, linewidth=2.5)

# sns.jointplot(x='height', y='length', data=df)
