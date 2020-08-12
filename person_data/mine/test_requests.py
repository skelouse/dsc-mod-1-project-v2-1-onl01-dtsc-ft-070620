"""Code written by Sam Stoltenberg
https://github.com/skelouse"""
import re
import json
import requests
from bs4 import BeautifulSoup
file_name = 'sams.txt'
data = {}

def extract_budget(string):
    """Extracts the numeric from the budget"""
    try:
        return int("".join(re.findall(r'\b\d+\b', string)))
    except ValueError:
        return None

def extract_numbers(string):
    """Extracts the numeric from the string with numbers"""
    try:
        return int("".join(re.findall(r'\b\d+\b', string)))
    except ValueError:
        return None

usd_from_GBP = 1.31
usd_from_NOK = 0.11
usd_from_DKK = 0.16
usd_from_AUD = 0.71
usd_from_EUR = 1.18
usd_from_MXN = 0.045

def find(soup, q_num):
    """Update data dictionary of given q_num using soup"""
    budget = None
    gross = None
    ww_gross = None
    rating = None
    # Find the rating
    for div in soup.findAll('div', class_='subtext'):
        rating = (div.text.split('\n')[1].replace(' ', '').replace('\n', ''))
    # Find the budget, gross, ww_gross in page
    for h4 in soup.findAll('h4'):
        if h4.text.startswith('Budget'):
            text = h4.parent.text
            if 'GBP' in text:
                text = text.split(' ')[0].replace('GBP', '')
                budget = int(extract_budget(text)*usd_from_GBP)
            elif 'NOK' in text:
                text = text.split(' ')[0].replace('NOK', '')
                budget = int(extract_budget(text)*usd_from_NOK)
            elif 'DKK' in text:
                text = text.split(' ')[0].replace('DKK', '')
                budget = int(extract_budget(text)*usd_from_DKK)
            elif 'AUD' in text:
                text = text.split(' ')[0].replace('AUD', '')
                budget = int(extract_budget(text)*usd_from_AUD)
            elif 'EUR' in text:
                text = text.split(' ')[0].replace('EUR', '')
                budget = int(extract_budget(text)*usd_from_EUR)
            elif 'MXN' in text:
                text = text.split(' ')[0].replace('MXN', '')
                budget = int(extract_budget(text)*usd_from_MXN)
            elif '$' in text:
                text = text.split(' ')[0]
                budget = extract_budget(text)
            else:
                print('failed', text)
            
        elif h4.text.startswith('Gross USA'):
            text = h4.parent.text
            text = text.split(' ')[2]
            gross = extract_numbers(text)
            
        elif h4.text.startswith('Cumulative Worldwide'):
            text = h4.parent.text
            text = text.split(' ')[3]
            ww_gross = extract_numbers(text)
    if budget or gross or ww_gross or rating:
        new_data = {
            q_num:{
                'budget': budget,
                'gross': gross,
                'ww_gross': ww_gross,
                'rating': rating
                }
            }
        data.update(new_data)

url = "https://www.imdb.com/title/"
def get_soup(q_num):
    """Gets the beautiful soup soup for a given q_num"""
    req = requests.get(str(url+q_num))
    return BeautifulSoup(req.content.decode(), 'html.parser')

def save(data):
    """Save the data"""
    with open('data.json', 'w') as f:
        json.dump(data, f)

with open(file_name, 'r') as f:
    q_list = f.read().split(',')


x = 0
for num in q_list:
    x += 1
    if x == 100:
        x = 0
        save(data)
    print('analysing', num)
    find(get_soup(num), num)

save(data)

