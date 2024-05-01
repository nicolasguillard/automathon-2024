import os
import csv

# read ids from dataset.csv
def read_ids():
    ids  = {}
    with open('dataset.csv', 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            ids[row[0]] = row[1]
    return ids

# take approx 10% of the data from train

import json

train_dir = os.path.expanduser('~/automathon-2024-dataset/train_dataset')
test_dir = os.path.expanduser('~/automathon-2024-dataset/test_dataset')

meta_train = json.load(open(os.path.join(train_dir, 'metadata.json')))
meta_test = json.load(open(os.path.join(test_dir, 'metadata.json')))

for i, key in enumerate(meta_train.keys()):
    if i % 10 == 0:
        meta_test[key] = meta_train[key]
        del meta_train[key]
        os.system(f'mv {os.path.join(train_dir, key)} {os.path.join(test_dir, key)}')

json.dump(meta_train, open(os.path.join(train_dir, 'metadata.json'), 'w'))
json.dump(meta_test, open(os.path.join(test_dir, 'metadata.json'), 'w'))
