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
	with open('chitai-gorod_dataset.csv', mode='a', newline='', encoding='utf-16') as f:
		#newline - to avoid blank rows after each record
		#encoding utf-16 - we are in russia, thats all
		writer = csv.writer(f, delimiter=';')
		writer.writerow(data)
write_csv(['publisher', 'year', 'pages', 'height', 'lenght', 'width', 'cover_type', 'edition', 'weight', 'book_pg', 'title', 'author', 'genre', 'price'])

for number in range(0, 99999):
	url = 'https://www.chitai-gorod.ru/catalog/book/11{:05d}'.format(number)

	try:
		print(url + ' code: ' + str(url_code(url)))
		
		if url_code(url) == 200:
			soup = BS(get_html(url), 'lxml')

			try:
				prod_props = soup.find_all('div', {'class': 'product-prop'})

				#initial variable blank
				publisher, year, pages, height, lenght, width, cover_type, edition, weight, book_pg, title, author, price, genre = \
				'', '', '', '', '', '', '', '', '', '', '', '', '', ''

				for prop in prod_props:

					if prop.find('div', {'class': 'product-prop__title'}).get_text(strip=True) == 'Издательство':
						publisher = prop.find('div', {'class': 'product-prop__value'}).get_text(strip=True)
						print('Издатель: ', publisher)

					if prop.find('div', {'class': 'product-prop__title'}).get_text(strip=True) == 'Год издания':
						year = prop.find('div', {'class': 'product-prop__value'}).get_text(strip=True)
						print('Год: ', year)

					if prop.find('div', {'class': 'product-prop__title'}).get_text(strip=True) == 'Кол-во страниц':
						pages = prop.find('div', {'class': 'product-prop__value'}).get_text(strip=True)
						print('Страниц: ', pages)

					if prop.find('div', {'class': 'product-prop__title'}).get_text(strip=True) == 'Формат':
						dimensions = prop.find('div', {'class': 'product-prop__value'}).get_text(strip=True)
						height = dimensions.split(' x ')[0]
						lenght = dimensions.split(' x ')[1]
						width = dimensions.split(' x ')[2]
						print('Размеры: высота {} см. ширина {} см. толщина {} см.'.format(height, lenght, width))

					if prop.find('div', {'class': 'product-prop__title'}).get_text(strip=True) == 'Тип обложки':
						cover_type = prop.find('div', {'class': 'product-prop__value'}).get_text(strip=True)
						print('Тип обложки: ', cover_type)

					if prop.find('div', {'class': 'product-prop__title'}).get_text(strip=True) == 'Тираж':
						edition = prop.find('div', {'class': 'product-prop__value'}).get_text(strip=True)
						print('Тираж: ', edition)

					if prop.find('div', {'class': 'product-prop__title'}).get_text(strip=True) == 'Вес, г':
						weight = prop.find('div', {'class': 'product-prop__value'}).get_text(strip=True)
						print('Вес: ', weight)

					if prop.find('div', {'class': 'product-prop__title'}).get_text(strip=True) == 'Возрастные ограничения':
						book_pg = prop.find('div', {'class': 'product-prop__value'}).get_text(strip=True)
						print('Возрастные ограничения: ', book_pg)
				title = soup.find('h1', {'class': 'product__title js-analytic-product-title'}).get_text(strip=True)
				print(title)
				author = soup.find('a', {'class': 'link product__author'}).get_text(strip=True)
				print(author)
				price = soup.find('div', {'class': 'price'}).get_text(strip=True).split()[0]
				print(price)
				genre = soup.find_all('li', {'class': 'breadcrumbs__item'})[-1].get_text(strip=True)
				print('Genre: ', genre)

				write_csv([publisher, year, pages, height, lenght, width, cover_type, edition, weight, book_pg, title, author, genre, price])	

			except:
				pass
	except:
		pass
		