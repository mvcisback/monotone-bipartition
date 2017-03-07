from collections import namedtuple

import funcy as fn
import numpy as np
from lenses import lens

import multidim_threshold as mdt

ProjVec = namedtuple('ProjVec', 'root direc')


def new_hi(pi, hi):
    root, direc = mdt.map_array(pi)
    v = ((np.array(hi) - root) / direc).min()
    return root + v * direc


def learn_diagsearch(oracle, bot):
    bot_val = oracle(bot)
    return mdt.binsearch if isinstance(bot_val, bool) else mdt.weightedbinsearch


def project(hi, member_oracles, proj, *, searches=None):
    rec = mdt.to_rec(lo=proj.root, hi=new_hi(proj, hi))
    if searches is None:
        searches = (learn_diagsearch(f, proj.root) for f in member_oracles)

    return [search(rec, oracle) for search, oracle in zip(searches, member_oracles)]


proj_key = fn.compose(tuple, mdt.map_tuple)


def projections(hi, member_oracles, proj_vectors):
    projections = (project(hi, member_oracles, vec) for vec in proj_vectors)
    return {proj_key(vec): proj for vec, proj in zip(proj_vectors, projections)}


def generate_proj_vecs(lo, hi, direc=None):
    if direc is None:
        direc = hi - lo

    vecs = [ProjVec(lo, direc)]
    while True:
        yield from vecs
        vecs = [ProjVec(r, direc) for r in next_roots(lo, hi, vecs)]


def project_along_axes(lo, mid):
    return [lens(lo)[i].set(mid[i]) for i in range(len(lo))]


def midpoint(lo, hi, proj_vec):
    hi = new_hi(proj_vec, hi)
    return (np.array(lo) + np.array(hi)) / 2


def next_roots(lo, hi, prev_vecs):
    mids = [midpoint(v.root, hi, v) for v in prev_vecs]
    return fn.cat(project_along_axes(lo, mid) for mid in mids)
