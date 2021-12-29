

import sys
import numpy as np

import utils
from model import LogisticRegression, DNN, RankNet, LambdaRank
from prepare_data import label_file_pat, group_file_pat, feature_file_pat

def load_data(type):