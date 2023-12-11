import requests
import os

if not os.path.exists('./data/benchmarks/squad'):
    os.makedirs('./data/benchmarks/squad')
    os.makedirs('./data/outputs')

url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
files = ['train-v2.0.json', 'dev-v2.0.json']

for file in files:
    req = requests.get(f'{url}{file}')
    # write file
    with open(f'./data/benchmarks/squad/{file}', 'wb') as f:
        for chunk in req.iter_content(chunk_size=5):
            f.write(chunk)