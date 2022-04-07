import json, os, re
from pprint import pprint

'''
- lecture du premier tweet

tips:
- users.docx => petite liste de users annotée
'''

rt=True
ROOT='/data/datasets/elyzee_2022/10:01:2021-00:00:00_03:22:2022-00:00:00'
files = os.listdir(ROOT)
for file in files:
    with open(f'{ROOT}/{file}', 'r') as f:
        tweet = json.load(f)
        if not re.search('^RT @(.*?):\s', tweet.get('text')) or rt:
            print(file)
            print('==='*10)
            pprint(tweet)
            print()
            # print(re.findall(r'^RT @(.*?):\s', tweet.get('text'))[0])
            print('Print the next one ? y/n ')
            if input() != 'y':
                break

