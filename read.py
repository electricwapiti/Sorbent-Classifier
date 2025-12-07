# Attempt to understand data file in order to replicate it

import gzip
import pickle

with gzip.open("sorbents.pkl.gz", "rb") as f:
    data = pickle.load(f, encoding="latin1")

train, val, test = data
print(len(train[0]), len(val[0]), len(test[0]))

#import pprint
#pprint.pprint(data)

print(data[0][0][:5])   # First 5 training images (flattened)
print(data[0][1][:5])   # First 5 training labels