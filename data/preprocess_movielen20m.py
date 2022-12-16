import os.path
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from gen_sequence import split_train_val_test_by_time

def gen_ml20_data():
    path = "ml20_time/"
    if not os.path.exists(path):
        os.mkdir(path)
    cols = []




if __name__ == '__main__':
    gen_ml20_data()