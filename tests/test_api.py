import re
import json
import time
import requests
import threading
from bs4 import BeautifulSoup
import numpy as np
np.NaN
url = "https://www.imdb.com/title/"
data = {}
cont = [True]
def parse(q_num, *args):
    req = requests.get(url=(url+q_num))
    if req.status_code == 200:
        try:
            # use soup instead
            string = req.content.decode()
            budget_string = string[string.index('Budget'): string.index('estimated')]
            budget = int("".join(re.findall(r'\b\d+\b', budget_string)))

            gross_string = string[string.index('Gross USA'): string.index('Cumulative')]
            gross = int("".join(re.findall(r'\b\d+\b', gross_string)))

            ww_gross_string = string[string.index('Worldwide Gross'): string.index('Worldwide Gross')+50]
            ww_gross = int("".join(re.findall(r'\b\d+\b', ww_gross_string)))
            data[q_num] = {
                'budget': budget,
                'gross': gross,
                'ww_gross': ww_gross
            }
        except ValueError:
            pass
        return True
    else:
        cont[0] = False
        return False


with open('qlist.txt', 'r') as f:
    q_list = f.read().split(',')


y = 0
x = 0
for q_num in q_list:
    if not cont[0]:
        break
    print("analysing", q_num)
    parse(q_num)
    x += 1
    if x == 100:
        x = 0
        y += 1
        with open(str('./data/data%s.json'%y), 'w') as f:
            json.dump(data, f)







