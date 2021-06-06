import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
from naive_bayes import *


data = pd.read_csv('data.txt', header=None, sep=',')
# print(data)
X = data[data.columns[0:2]]
y = data[data.columns[2]]
clf = NB(1)
clf.fit(X, y)