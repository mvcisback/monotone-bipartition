import numpy as np
import multidim_threshold
from collections import namedtuple
import more_itertools

project_along = namedtuple('project_along', 'root direc')


new_hi = lambda pi, hi: pi.root + np.multiply(pi.direc, np.true_divide(np.array(hi) - np.array(pi.root), np.array(pi.direc)).min())
midpoint = lambda a, b: np.true_divide(np.array(a) + np.array(b), 2)

def project_along_axes(lo, mid):
    return [project_along_i(lo[:], mid, i) for i in range(len(lo))]

def project_along_i(lo, mid, i):
    lo[i] = mid[i]
    return lo


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

def generate_lambdas(lo, hi, direc, k):
    lambda_0 = project_along(lo, direc)
    lambdas = [lambda_0]

    new_high = new_hi(lambda_0, hi)
    lambda_mid = [midpoint(lo, new_high)]
    iter = 0

    while iter < k:
        projected_points = find_projected_points(lo, lambda_mid)
        projected_points = list(more_itertools.flatten(projected_points))
        lambda_mid = []
        for i in range(len(projected_points)):
            lambdas.append(project_along(projected_points[i], direc))
            lambda_mid.append(midpoint(projected_points[i], new_hi(lambdas[-1], hi)))
        iter = iter + 1
    return lambdas

def find_projected_points(lo, lambda_mid):
    return [project_along_axes(lo, lambda_mid[i]) for i in range(len(lambda_mid))]












