
import os
import numpy as np


label_file_pat = "../data/processed/%s_label.npy"
group_file_pat = "../data/processed/%s_group.npy"
feature_file_pat = "../data/processed/%s_feature.npy"


def convert(type):
    data_path = os.path.join("..", "data/MQ2008/Fold1/"+ type + ".txt")

    labels = []
    features = []
    groups = []
    with open(data_path, "r") as f:
        for line in f:
        