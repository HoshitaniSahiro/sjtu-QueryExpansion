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


def example_list2dict(input):
    output = dict()
    for word in input.split():
        if output.get(word) is None:
            output[word] = 0
        output[word] += 1
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


def bm25(query, doc, idf, avg_doc_len=374):
    k1 = 2.0
    k2 = 1.0
    b = 0.75
    score = 0.0
    for word in query:
        if doc.get(word) == None:
            continue
        W_i = idf[word]
        f_i = doc[word]
        qf_i = query[word]
        doc_len = sum(doc.values())
        K = k1 * (1 - b + b * doc_len / avg_doc_len)
        R1 = f_i * (k1 + 1) / (f_i + K)
        R2 = qf_i * (k2 + 1) / (qf_i + k2)
        R = R1 * R2
        score += W_i * R
    return score


def docs2vecs(doc_dict):
    vectors = {}
    for doc_id, doc in doc_dict.items():
        words = str(doc)
        vectors[doc_id] = Counter()
        for word in words.split():
            vectors[doc_id][word] += 1

    for doc_id in vectors:
        for word in vectors[doc_id]:
            vectors[doc_id][word] *= idf[word]

    return vectors


def get_candidates(query, add_doc_dict, all_doc_dict):
    candidates = []
    features = {}
    tf = {}
    # calculate term frequency
    for doc_id, doc in add_doc_dict.items():
        words = str(doc)
        tf[doc_id] = Counter()
        for word in words.split():
            tf[doc_id][word] += 1
    # calculate tfidf
    tfidf = dict(tf)
    for doc_id in tfidf:
        for word in tfidf[doc_id]:
            tfidf[doc_id][word] *= idf[word]
    for doc_id in tfidf:
        candidates.extend(tfidf[doc_id].most_common(30))
    for i in range(len(candidates)):
        candidates[i] = candidates[i][0]
    candidates = list(set(candidates))

    # f1: term distribution in relevant docs / remove noise
    n_all_words = 0
    for doc_id, doc in add_doc_dict.items():
        words = str(doc)
        for word in words.split():
            n_all_words += 1
    trim_candidates = list(candidates)
    for candidate in candidates:
        count = 0
        for doc_id, doc in add_doc_dict.items():
            words = str(doc)
            if candidate in words.split():
                count += 1
        if count < 3:
            trim_candidates.remove(candidate)
            if candidate in features:
                features.pop(candidate)
        else:
            f1 = math.log(count / n_all_words)
            features[candidate] = [f1]
    candidates = trim_candidates

    # f2: term distribution in the whole doc collection
    n_all_words = 0
    for doc_id, doc in all_doc_dict.items():
        words = str(doc)
        for word in words.split():
            n_all_words += 1
    for candidate in candidates:
        count = 0
        for doc_id, doc in all_doc_dict.items():
            words = str(doc)
            if candidate in words.split():
                count += 1
        f2 = math.log(count / n_all_words)
        features[candidate].append(f2)

    # f3: co-occurence with query terms in relevant docs
    for candidate in candidates:
        f3 = 0
        for doc_id in add_doc_dict:
            if candidate in add_doc_dict[doc_id]:
                for i in query.split():
                    if i in add_doc_dict[doc_id]:
                        f3 += 1
        f3 = math.log(f3 + 0.5)
        features[candidate].append(f3)

    # f3: co-occurence with query terms in the whole doc collection
    for candidate in candidates:
        f4 = 0
        for doc_id in all_doc_dict:
            if candidate in all_doc_dict[doc_id]:
                for i in query.split():
                    if i in all_doc_dict[doc_id]:
                        f4 += 1
        f4 = math.log(f4 + 0.5)
        features[candidate].append(f4)

    return candidates, features


def get_F(query_id, result_train, rele_q):
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    for e in rele_q:
        if e[2] == '1':
            if [query_id, e[1]] in result_train:
                tp += 1
            else:
                fn += 1
        if e[2] == '0':
            if [query_id, e[1]] in result_train:
                fp += 1
            else:
                tn += 1
    if tp == 0:
        acc = 0
        rec = 0
        F = 0.01
    else:
        acc = tp / (tp + fp)
        rec = tp / (tp + fn)
        F = 2 * acc * rec / (acc + rec)

    return F


def train():
    # Read Files and Calculate IDF
    doc_dict = read_file("./doc.txt")
    print("Docs Read.")
    query_dict = read_file("./query.txt")
    print("Queries Read.")
    global idf
    idf = joblib.load("./idf.pkl")
    print("IDF Loaded.")
    read_rele = open("./rele.txt")
    rele = []
    for i in range(199):
        rele.append([])
    for line in read_rele.readlines():
        query_id, doc_id, flag = line.strip().split('\t')
        query_id_int = int(query_id) - 50
        rele[query_id_int].append([query_id, doc_id, flag])
    print("Relevance Read.")

    # Sort Train and Test Sets
    scores_0 = joblib.load("./scores_0.pkl")
    scores_0_sorted = []
    for q in range(0, 249):
        scores_0_sorted.append(
            sorted(scores_0[q], key=lambda x: x[2], reverse=True))
    scores_0 = scores_0_sorted

    # Generate Train Dataset
    scores_1 = []
    k = 20
    r = 30
    x = np.empty((0, 4))
    y = np.empty((0))
    for q in tqdm(range(50, 249)):
        query_id = str(q)
        query_this = query_dict[query_id]
        add_doc_dict = {}
        all_doc_dict = {}
        scores_1.append([])
        # calculate old result
        result_train_0 = []
        for i in range(r):
            result_train_0.append([query_id, scores_0[q][i][1]])
        F_0 = get_F(query_id, result_train_0, rele[q - 50])
        # relevent (top k) docs
        for i in range(k):
            add_doc_id = scores_0[q][i][1]
            add_doc_dict[add_doc_id] = doc_dict[add_doc_id]
        # all docs for current query
        for e in range(300):
            doc_id = str(q) + '_' + str(e)
            if doc_dict.get(doc_id) == None:
                break
            else:
                all_doc_dict[doc_id] = doc_dict[doc_id]
        candidates, features = get_candidates(query_this, add_doc_dict,
                                              all_doc_dict)
        # every candidate
        candidates_contribute = []
        for cnt in range(len(candidates)):
            candidate = candidates[cnt]
            # refresh query
            new_query_v = Counter(query_this.split())
            if new_query_v.get(candidate) == None:
                new_query_v[candidate] = 0
            new_query_v[candidate] += 0.5
            # calculate and sort scores
            for e in range(300):
                doc_id = str(q) + '_' + str(e)
                if doc_dict.get(doc_id) == None:
                    break
                else:
                    doc_this = example_list2dict(doc_dict[doc_id])
                    score_this = bm25(new_query_v, doc_this, idf)
                    scores_1[q - 50].append([query_id, doc_id, score_this])
            scores_1[q - 50].sort(key=lambda x: x[2], reverse=True)
            # get result for this new query
            result_train_1 = []
            for i in range(r):
                result_train_1.append([query_id, scores_1[q - 50][i][1]])
            # calculate the change to F1
            F_1 = get_F(query_id, result_train_1, rele[q - 50])
            # if F_0 == 0:
            #     candidates_contribute.append(((candidate, F_1)))
            candidates_contribute.append((candidate, F_1 - F_0))

        # Sort and find the best and worst candidates
        candidates_contribute = sorted(candidates_contribute, key=lambda x : x[1], reverse=True)
        x_q = []
        for i in range(30):
            x_q.append(features[candidates_contribute[i][0]])
            y = np.append(y, 1)
        x = np.row_stack((x, ppc.scale(x_q))) # normalization
        x_q = []
        for i in range(30):
            leng = len(candidates_contribute)
            x_q.append(features[candidates_contribute[leng - i - 1][0]])
            y = np.append(y, 0)
        x = np.row_stack((x, ppc.scale(x_q))) # normalization

    # Train SVM Model
    print("SVM Training...")
    model = svm.SVC(class_weight='balanced')
    model.fit(x, y)
    print("Training Completed!")

    # On Test Dataset
    scores_1 = []
    result_test = []
    for q in tqdm(range(0, 50)):
        x_q = []
        query_id = str(q)
        query_this = query_dict[query_id]
        add_doc_dict = {}
        all_doc_dict = {}
        scores_1.append([])
        # relevent (top k) docs
        for i in range(k):
            add_doc_id = scores_0[q][i][1]
            add_doc_dict[add_doc_id] = doc_dict[add_doc_id]
        # all docs for current query
        for e in range(300):
            doc_id = str(q) + '_' + str(e)
            if doc_dict.get(doc_id) == None:
                break
            else:
                all_doc_dict[doc_id] = doc_dict[doc_id]
        candidates, features = get_candidates(query_this, add_doc_dict, all_doc_dict)
        # decide every candidate
        new_query_v = Counter(query_this.split())
        for candidate in candidates:
            x_q.append(features[candidate])
        x_q_scaled = np.empty((0, 4))
        x_q_scaled = np.row_stack((x_q_scaled, ppc.scale(x_q)))
        pred = model.predict(x_q_scaled)
        for i in range(len(candidates)):
            if pred[i]:
                # refresh query
                candidate = candidates[i][0]
                if new_query_v.get(candidate) == None:
                    new_query_v[candidate] = 0
                new_query_v[candidate] += 0.5
        # calculate and sort scores
        for e in range(300):
            doc_id = str(q) + '_' + str(e)
            if doc_dict.get(doc_id) == None:
                break
            else:
                doc_this = example_list2dict(doc_dict[doc_id])
                score_this = bm25(new_query_v, doc_this, idf)
                scores_1[q].append([query_id, doc_id, score_this])
        scores_1[q].sort(key=lambda x: x[2], reverse=True)
        # get result for this new query
        for i in range(len(scores_1[q])):
            result_test.append([query_id, scores_1[q][i][1]])
    
    # Write Result
    with open("result_ml.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["QueryId", "DocumentId"])
        writer.writerows(result_test)
    with open('result_ml.csv', 'rt') as fin:
        lines = ''
        for line in fin:
            if line != '\n':
                lines += line
    with open('result_ml.csv', 'wt') as fout:
        fout.write(lines)
    print("Result Written.")


if __name__ == '__main__':
    train()