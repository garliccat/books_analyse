import requests
from bs4 import BeautifulSoup as BS
import csv
import codecs


def url_code(url):
	response = requests.get(url)
	return response.status_code

def get_html(url):
	try:
		r = requests.get(url)
	except:
		print('unable to reach page ' + url)
		return None
	return r.text

def write_csv(data):
	with open('labirint_dataset.csv', mode='a', newline='', encoding='utf-16') as f:
		#newline - to avoid blank rows after each record
		#encoding utf-16 - we are in russia, thats all
		writer = csv.writer(f, delimiter=';')
		writer.writerow(data)
write_csv(['id', 'title', 'author', 'year', 'publisher', 'genre', 'pages', 'weight', 'height', 'lenght', 'width', 'price'])

# here comes a more handy approach. labirint has (god thank you) a simple indexing of all the books starting from 1 and to infinity.
# thus, i will try not die waiting and collect as much as possible.

for number in range(400000, 500000):
	url = 'https://www.labirint.ru/books/{}'.format(str(number))

	try:
		print(url)
		
		if url_code(url) == 200:
			soup = BS(get_html(url), 'lxml')

			try:
				title = soup.find('div', {'id': 'product-title'}).find('h1').get_text(strip=True)
				print('Название: ', title)
			except:
				title = ''

			try:
				author = soup.find('div', {'class': 'authors'}).find('a').get_text(strip=True)
				print('Автор: ', author)
			except:
				author = ''

			try:
				text = soup.find('div', {'class': 'publisher'}).get_text(strip=True)
				year = text.split()[-2]
				publisher = text.split(':')[1].split(',')[0]
				print('Год: ', year)
				print('Издательство: ', publisher)
			except:
				year = ''
				publisher = ''

			try:
				genre = soup.find('div', {'class': 'genre'}).find('a').get_text(strip=True)
				print('Жанр: ', genre)
			except:
				genre = ''

			try:
				pages = soup.find('div', {'class': 'pages2'}).get_text(strip=True).split()[1]
				print('Страниц: ', pages)
			except:
				pages = ''

			try:
				weight = soup.find('div', {'class': 'weight'}).get_text(strip=True).split()[1]
				print('Вес: ', weight)
			except:
				weight = ''

			try:
				text = soup.find('div', {'class': 'dimensions'}).get_text(strip=True)
				height = text.split()[1].split('x')[0]
				lenght = text.split()[1].split('x')[1]
				width = text.split()[1].split('x')[2]
				print('Размеры: высота - ', height, ' мм. ширина - ', lenght, ' мм. толщина - ', width, ' мм.')
			except:
				height, width, lenght = '', '', ''


			try:
				price = soup.find('span', {'class': 'buying-price-val-number'}).get_text(strip=True)
				print('Цена: ', price)
			except:
				price = ''

			print('ID: ', number)

			print()

			write_csv([number, title, author, year, publisher, genre, pages, weight, height, lenght, width, price])	

	except:
		pass
		