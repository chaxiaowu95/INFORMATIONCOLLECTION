

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
    # keep in mind the followings:
    # 1. batch size should be large enough to ensure there are samples of different
    # relevance labels from the same group, especially when you use "sample" as "batch_sampling_method"
    # this ensure the gradients are nonzeros and stable across batches,
    # which is important for pairwise method, e.g., ranknet and lambdarank
    # 2. batch size should not be very large since the lambda_ij matrix in ranknet and lambdarank