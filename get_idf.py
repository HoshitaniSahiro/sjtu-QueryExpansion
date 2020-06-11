import numpy as np
import joblib
import math
import csv
from collections import Counter
from tqdm import tqdm


def read_file(file):
    fp = open(file)
    output = dict()
    for line in fp.readlines():
        id, text = line.strip().split('\t')
        output[id] = text
    return output


def cal_idf(doc_dict):
    doc_num = len(doc_dict)
    idf = dict()
    for doc_id in doc_dict:
        doc_text = list(set(doc_dict[doc_id].split()))
        for word in doc_text:
            if idf.get(word) is None:
                idf[word] = 0
            idf[word] += 1
    for word in idf:
        idf[word] = math.log((doc_num - idf[word] + 0.5) / (idf[word] + 0.5))
    return idf


if __name__ == '__main__':
    doc_dict = read_file("./doc.txt")
    idf = cal_idf(doc_dict=doc_dict)
    joblib.dump(idf, "./idf.pkl")
    print("IDF Saved.")