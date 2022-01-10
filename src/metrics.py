

import numpy as np


# taken from: https://github.com/andreweskeclarke/learning-rank-public
def calc_err(predicted_order):
    err = 0
    prev_one_min_rel_prod = 1
    previous_rel = 0
    T = len(predicted_order) if len(predicted_order) < 10 else 10
    for r in range(T):
        rel_r = calc_ri(predicted_order, r)
        one_min_rel_prod = (1 - previous_rel) * prev_one_min_rel_prod
        err += (1 / (r+1)) * rel_r * one_min_rel_prod
        prev_one_min_rel_prod = one_min_rel_prod
        previous_rel = rel_r

    return err
