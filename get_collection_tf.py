import numpy as np
import joblib
import pickle
import math
import csv
from collections import Counter
from sklearn import svm
from sklearn import preprocessing as ppc
from tqdm import tqdm


def read_file(file):
    fp = open(file)
    output = dict()
    for line in fp.readlines():
        id, text = line.strip().split('\t')
        output[id] = text
    return output

doc_dict = read_file("./doc.txt")
print("Docs Read.")
tf = Counter()
n_all_words = 0
for doc_id, doc in tqdm(doc_dict.items()):
    words = str(doc)
    for word in words.split():
        n_all_words += 1
        if word in tf == False:
            tf[word] = 0
        tf[word] += 1
for word in tf:
    tf[word] = math.log(tf[word] / n_all_words)

joblib.dump(tf, "collection_tf.pkl")
print("done")
