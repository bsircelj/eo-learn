# Za vsak label pogleda, kateri je večinski razred in ga smatra za pravilnega. Izračuna razmerje med pravilnim in
# nepravilnim uteženo z velikostjo patcha
from collections import Counter
import numpy as np
from functools import reduce


def evaluate_clusters(predicted, truth, no_clusters=None):
    if no_clusters is None:
        no_clusters = np.amax(predicted)

    predicted = np.ravel(predicted)
    truth = np.ravel(truth)
    both = [(p, t) for p, t in zip(predicted, truth)]
    score = dict.fromkeys(range(no_clusters + 1), None)

    def each_pix(acc, current_pix):
        pred, tru = current_pix
        if pred == 0:
            return acc
        if acc[pred] is None:
            acc[pred] = [tru]
        else:
            acc[pred].append(tru)
        return acc

    score = reduce(each_pix, both, score)

    def count(acc, table):
        if table is None:
            return acc
        area = Counter(table)
        _, st = area.most_common(1)[0]
        return acc + st

    total_corr = reduce(count, score.values(), 0)
    total_size = predicted.size - sum(predicted == 0)

    return total_corr / total_size


'''
correct_pixels = 0
for p in range(1, no_clusters):
    # func = lambda a, b=[]: b.extend(a[1]) if a[0] == p else b
    # underlying = reduce(func, both)

    underlying = []
    for pred_pix, tr_pix in zip(predicted, truth):
        if pred_pix == p:
            if underlying == []:
                underlying = [tr_pix]
            else:
                #underlying = np.extend(underlying, tr_pix)
                underlying.append(tr_pix)

    if not underlying:
        continue

    area = Counter(underlying)
    c = area.most_common(1)[0]
    #print(underlying)
    #print(c)
    _, st = c
    correct_pixels += st
    '''
