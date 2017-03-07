from collections import namedtuple

import funcy as fn
import numpy as np
from lenses import lens

import multidim_threshold as mdt

ProjVec = namedtuple('ProjVec', 'root direc')


def clip_rec(pi, hi):
    root, direc = mdt.map_array(pi)
    v = ((np.array(hi) - root) / direc).min()
    return root + v * direc


def learn_search(oracle, bot):
    kind = oracle(bot)
    search = mdt.binsearch if isinstance(kind, bool) else mdt.weightedbinsearch
    return fn.partial(search, oracle=oracle)


def projections(hi, proj, *, searches):
    rec = mdt.to_rec(lo=proj.root, hi=clip_rec(proj, hi))
    return [search(rec) for search in searches]


def generate_projections(lo, hi, member_oracles, *, direc=None, searches=None):
    proj_vecs = generate_proj_vecs(lo, hi, direc)
    if searches is None:
        searches = [learn_search(f, lo) for f in member_oracles]

    for vec in proj_vecs:
        yield projections(hi, vec, searches=searches)


def generate_proj_vecs(lo, hi, direc=None):
    lo, hi = mdt.map_array((lo, hi))
    if direc is None:
        direc = hi - lo

    vecs = [ProjVec(lo, direc)]
    while True:
        yield from vecs
        vecs = [ProjVec(r, direc) for r in next_roots(lo, hi, vecs)]


def project_along_axes(lo, mid):
    return [lens(lo)[i].set(mid[i]) for i in range(len(lo))]


def midpoint(lo, hi, proj_vec):
    hi = clip_rec(proj_vec, hi)
    return (np.array(lo) + np.array(hi)) / 2


def next_roots(lo, hi, prev_vecs):
    mids = [midpoint(v.root, hi, v) for v in prev_vecs]
    return fn.cat(project_along_axes(lo, mid) for mid in mids)
