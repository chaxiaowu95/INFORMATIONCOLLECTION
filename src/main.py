

import sys
import numpy as np

import utils
from model import LogisticRegression, DNN, RankNet, LambdaRank
from prepare_data import label_file_pat, group_file_pat, feature_file_pat

def load_data(type):

    labels = np.load(label_file_pat%type)
    qids = np.load(group_file_pat % type)
    features = np.load(feature_file_pat%type)

    X = {
        "feature": features,
        "label": labels,
        "qid": qids
    }
    return X


utils._makedirs("../logs")
logger = utils._get_logger("../logs", "tf-%s.log" % utils._timestamp())

params_common = {
    # you might have to tune the batch size to get ranknet and lambdarank working