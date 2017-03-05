import numpy as np
import multidim_threshold
from collections import namedtuple, Iterable

project_along = namedtuple('project_along', 'root direc')


new_hi = lambda pi, hi: pi.root + np.multiply(pi.direc, np.true_divide(np.array(hi) - np.array(pi.root), np.array(pi.direc)).min())

def project_singleLambda(hi, is_member, pi):
    new_high = new_hi(pi, hi)
    rec = multidim_threshold.to_rec(pi.root, new_high)
    map = {}
    for i in range(len(is_member)):
        bool_oracle = isinstance(is_member[i](rec.bot), bool)
        diagsearch = multidim_threshold.binsearch if bool_oracle else multidim_threshold.weightedbinsearch
        low, mid, high = diagsearch(rec, is_member[i])
        map["trace" + str(i)] = mid
    return map

def projections(hi, is_member, pi_list):
    projection_dict = {}
    for i in range(len(pi_list)):
        map_i = project_singleLambda(hi, is_member, pi_list[i])
        projection_dict[tuple(map(tuple, pi_list[i]))] = map_i
    return projection_dict




