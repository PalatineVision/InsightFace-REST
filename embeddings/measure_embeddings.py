from save_utils import read_mat
from sklearn.neighbors import KNeighborsClassifier
from pathlib import Path
import numpy as np
import json

# model suffix
MODEL_SUFFIX = ['_glintr100.bin', '_w600k_mbf.bin', '_w600k_r50.bin']
LFW_ROOT = '../lfw'
FOLDS = 10
NUM_IN_FOLD = 600
BINS = 1001

try:
    result = json.load(open('result.json'))
except:
    result = {}

def get_train_test(X, y, i):
    X_test = X[i,...]
    y_test = y[i,...]
    X_train = np.concatenate([X[1:i,...], X[i+1:,...]], axis=0).reshape(-1,512,2)
    y_train = np.concatenate([y[1:i,...], y[i+1:,...]], axis=0).flatten()
    return X_train, y_train, X_test, y_test

def path_to_emb(name, num, model_suffix):
    return Path(LFW_ROOT) / "results" / name / f'{name}_{num:04d}.jpg{model_suffix}'

def calc_thresh(a, b):
    am = np.median(a)
    bm = np.median(b)
    MAX = max(np.max(a), np.max(b))
    MIN = min(np.min(a), np.min(b))

    # to differ distribution that "lefter" than second
    if am < bm:
        l, r = a, b
    else:
        l, r = b, a
    
    bins = np.linspace(MIN, MAX, num=BINS) 
    lcd = np.cumsum(np.histogram(l, bins=bins)[0])
    rcd = np.cumsum(np.histogram(r, bins=bins)[0])
    ith = np.argmax(lcd - rcd)
    thresh = (bins[ith] + bins[ith + 1]) / 2
    return thresh
    

def main():
    for suffix in MODEL_SUFFIX:
        # прочитать пары для сравнения
        with open(f'{LFW_ROOT}/pairs.txt', 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
            X = []
            y = []
            for l in lines:
                sp = l.strip('\n').split('\t')
                if len(sp) == 3:
                    file1 = path_to_emb(sp[0], int(sp[1]), suffix)
                    file2 = path_to_emb(sp[0], int(sp[2]), suffix)
                    y.append(1)
                elif len(sp) == 4:
                    file1 = path_to_emb(sp[0], int(sp[1]), suffix)
                    file2 = path_to_emb(sp[2], int(sp[3]), suffix)
                    y.append(0)

                a, b = read_mat(open(file1,'rb')), read_mat(open(file2,'rb'))        
                X.append(np.concatenate([a,b], axis=1))

            # разбить на фолды 10х(300 match+300 m/match) 10x600x512x2
            X = np.array(X).reshape(FOLDS, NUM_IN_FOLD, 512, 2)
            y = np.array(y).reshape(FOLDS, NUM_IN_FOLD)

        scores = []
        # 10 folds
        for i in range(len(X)):
            X_train, y_train, X_test, y_test = get_train_test(X, y, i)
            matches = X_train[np.argwhere(y_train == 1).flatten()]
            res_mat = []
            for m in matches:
                res_mat.append(np.linalg.norm(m[:,0] - m[:,1]))
            res_mat = np.array(res_mat)
            not_matches = X_train[np.argwhere(y_train == 0).flatten()]
            res_not_mat = []
            for m in not_matches:
                res_not_mat.append(np.linalg.norm(m[:,0] - m[:,1]))
            res_not_mat = np.array(res_not_mat)

            thresh = calc_thresh(res_mat, res_not_mat)
            res_pred = []
            for t in X_test:
                res_pred.append(np.linalg.norm(t[:,0] - t[:,1]) < thresh)
            scores.append(np.sum(np.array(res_pred) == y_test) / NUM_IN_FOLD)
        res = np.mean(np.array(scores))
        
        result[suffix] = res
        json.dump(result, open('result.json', 'w'))


if __name__ == '__main__':
    main()