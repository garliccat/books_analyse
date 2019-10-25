'''
Этот анализ затевался изначально просто чтобы посчитать среднюю высоту книги для строительства (очередного) шкафчика.
Но раз парсить, то парсить всё что можно. Поэтому помимо габаритов книги были собраны еще кое-какие данные.
Парсера тут нет, датасет уже собран.
Данные были любезно предоставлены (нет) сайтом магазина Читай Город.
Парсился так-же еще и сайт магазина labirint.ru, но из-за его объёма (получилось почти 50000 записей) и наличием аномалий решено было его не использовать.
'''

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.linear_model import LinearRegression

df = pd.read_csv('chitai_gorod_dataset.csv', sep=';', encoding='utf-16')
# df = pd.read_csv('labirint_dataset.csv', sep=';', encoding='utf-16')

print('Количество записей: ', df.shape[0])

''' Так как записей много (более 10000) и там присутствуют НЕ книги почему-то тоже, то обрезаем аномалии по высоте книги и весу, которые попадают за 99ый и перед 1ый квантилем.
Попадаются всякие адские фолианты или гигантские календари (которые почему-то у читай-города в разделе "книги")
'''
df = df[(df['height'] < df['height'].quantile(.999)) & (df['height'] > df['height'].quantile(.001))]
df = df[(df['weight'] < df['weight'].quantile(.999)) & (df['weight'] > df['weight'].quantile(.001))]


''' Делим все книги на две категории, ибо как правило в шкафах делают полки двух размеров (те, которые дома а не в библиотеках).
Полки для большинства книг и полки для высоких, больших талмудов, коранов, библий и прочих антропологических атласов и энциклопедий рыбака.
'''

height = df['height'].dropna()
hist, bins = np.histogram(height, bins=2)
# print('Hist = ', hist)
# print('Proportion = ', hist[0] / hist[1])
# print('bins = ', bins)
print('Высота основных полок (плюс зазор 3 см.) = ', round(bins[np.where(hist == hist.max())[0]][0] + 3, 1))
print('Высота дополнительных (высоких) полок (плюс 3 см.) = ', round((height.max() + 3), 1))

shelves_prop = hist.max() / hist.min()
print('Пропорция количества полок обычные / высокие: {:.1f}'.format(shelves_prop))

ax1 = sns.distplot(height, bins=2)
text_y = ax1.get_lines()[0].get_data()[1].max()
text_x = ax1.get_lines()[0].get_data()[0]
text_x = text_x[np.where(ax1.get_lines()[0].get_data()[1] == text_y)][0]
ax1.annotate(s=format(text_x, '.1f'), xy=(text_x, text_y))
plt.xlabel('Высота книги (см.)')
plt.ylabel('Вероятность')
plt.title('Распределение высоты книг')
plt.tight_layout()
plt.show()


print('\nСамые плодовитые авторы:')
print(df.groupby(by='author').count()['title'].sort_values(ascending=False).head())
print('\nТоп годов по релизам:')
print(df.groupby(by='year').count()['title'].sort_values(ascending=False).head())
print('\nТоп книг по количеству страниц:')
pages = df.groupby(by='pages').count()['title'].sort_values(ascending=False)
print(pages.head())
page_conspiracy = df[df['pages'] == pages.index[0]].groupby(by='publisher').count()['title'].sort_values(ascending=False)
print('\n', page_conspiracy.head())
print('\nСамый ленивый издатель (большинство книг с одинаковым количеством страниц): ', page_conspiracy.index[0])

############ ONLY FOR CHITAI GOROD ############
print('\nВозрастные рейтинги книг:')
book_ratings = df.groupby(by='book_pg').count()['title'].sort_values(ascending=False)
# print(book_ratings)
ax = sns.barplot(book_ratings.index, book_ratings.values)
for p in ax.patches:
	ax.annotate(format(p.get_height(), '.0f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
plt.xlabel('Возрастной рейтинг книг')
plt.ylabel('Количество книг')
plt.title('Возрастные рейтинги книг')
plt.show()


print('\nСамые дорогие книги (топ 10):')
print(df.sort_values(by='price', ascending=False)[['author', 'title', 'price']].head(10))

'''
Посмотрим как зависит стоймость книги от её веса (очевидно что весьма прямая зависимость)
Обучаем линейный регрессор:
'''
# df = df.dropna()
model = LinearRegression(fit_intercept=True)
X = df['weight'].to_numpy()[:, np.newaxis]
y = df['price'].to_numpy()
model.fit(X, y)
# print(model.coef_, model.intercept_)
xfit = np.linspace(0, X.max() * 1.05)
Xfit = xfit[:, np.newaxis]
yfit = model.predict(Xfit)
print('Минимально возможная стоймость книги (при весе 0 гр.) = {:.1f}'.format(model.intercept_))
print('Если бы продавали на вес, то 1 гр. книг стоил бы (начиная с {:.1f} г.) = {:.1f} руб.'.format(model.intercept_, model.coef_[0]))

sns.scatterplot(x='weight', y='price', data=df)
sns.lineplot(x=xfit, y=yfit)
plt.xlabel('Вес')
plt.ylabel('Цена')
plt.title('Распределение веса / цены книг.')
plt.tight_layout()
plt.show()

'''Распределение цен на книги в общем.
Отмечена самая распространенная цена.
'''

prices = df['price'].to_numpy()
ax1 = sns.kdeplot(prices, shade=True)
text_y = ax1.get_lines()[0].get_data()[1].max()
text_x = ax1.get_lines()[0].get_data()[0]
text_x = text_x[np.where(ax1.get_lines()[0].get_data()[1] == text_y)][0]
ax1.annotate(s=format(text_x, '.1f'), xy=(text_x, text_y))
plt.xlabel('Цена')
plt.ylabel('Вероятность')
plt.title('Распределение цен на книги')
plt.show()
print('')
print('Самая популярная цена (+-) = {:.2f}'.format(text_x))
print('Медианная цена = ', np.median(prices))
print('Средняя цена = ', np.mean(prices))
