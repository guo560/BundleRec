import time
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve


def get_best_confusion_matrix(y_true, y_pre):
    fpr, tpr, thresholds = roc_curve(y_true, y_pre)
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    best_threshold = round(thresholds[index], ndigits=4)
    martrix = confusion_matrix(y_true, y_pre >= best_threshold)
    try:
        tn, fp, fn, tp = martrix.ravel()
    except:
        tn, fp, fn, tp = 0, 0, 0, martrix[0][0]
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    acc = (tp + tn) / (tn + fp + fn + tp)
    return martrix, {"recall": recall, "precision": precision, "acc": acc}


def timeit(func):
    """
    装饰器，计算函数执行时间
    """
    def wrapper(*args, **kwargs):
        time_start = time.time()
        result = func(*args, **kwargs)
        time_end = time.time()
        exec_time = time_end - time_start
        print("{function} exec time: {time}s".format(function=func.__name__, time=exec_time))
        return result
    return wrapper
