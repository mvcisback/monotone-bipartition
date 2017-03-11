from operator import itemgetter as ig
from collections import namedtuple

import funcy as fn
import numpy as np
from lenses import lens
import random

import multidim_threshold as mdt

ProjVec = namedtuple('ProjVec', 'root direc')


def clip_rec(pi, hi):
    root, direc = mdt.map_array(pi)
    mask = direc != 0
    v = ((hi - root)[mask] / direc[mask]).min()
    return root + v * direc


def learn_search(oracle, bot):
    kind = oracle(bot)
    search = mdt.binsearch if isinstance(kind, bool) else mdt.weightedbinsearch
    return fn.compose(ig(1), fn.partial(search, oracle=oracle))


def projections(hi, proj, searches):
    rec = mdt.to_rec(lo=proj.root, hi=clip_rec(proj, hi))
    return [search(rec) for search in searches]


def generate_boundary_approxes(lo, hi, member_oracles, **kwargs):
    boundaries = [set() for _ in member_oracles]
    for points in generate_projections(lo, hi, member_oracles, **kwargs):
        for b, p in zip(boundaries, points):
            if p is not None:
                b.add(tuple(p))
        yield boundaries


def generate_projections(lo, hi, member_oracles, *, direc=None, searches=None, random=False):
    if searches is None:
        searches = [learn_search(f, lo) for f in member_oracles]

    lo, hi = mdt.map_array((lo, hi))
    # TODO: this seems to be broken
    #hi = axes_intersects(lo, hi, searches if random else hi

    if direc is None:
        direc = hi - lo

    root_func = random_root if random else next_roots
    proj_vecs = generate_proj_vecs(lo, hi, direc, root_func)

    for vec in proj_vecs:
        yield projections(hi, vec, searches)


def next_roots(lo, hi, prev_vecs):
    mids = [midpoint(v.root, hi, v) for v in prev_vecs]
    return fn.cat(project_along_axes(lo, mid) for mid in mids)


def generate_proj_vecs(lo, hi, direc=None, next_roots=next_roots):
    vecs = [ProjVec(lo, direc)]
    while True:
        yield from vecs
        vecs = [ProjVec(r, direc) for r in next_roots(lo, hi, vecs)]


def axes_intersects(lo, hi, searches):
    intersects = lambda b: b * \
        np.array(projections(hi, ProjVec(lo, b), searches))
    return sum(i.min(axis=0) for i in map(intersects, mdt.basis_vecs(len(lo))))


def project_along_axes(lo, mid):
    return [lens(lo)[i].set(mid[i]) for i in range(len(lo))]


def midpoint(lo, hi, proj_vec):
    return (lo + clip_rec(proj_vec, hi)) / 2


def random_root(lo, hi, vecs):
    dim = len(lo)
    t_proj = lambda lo, hi, i: lo[i] + (hi[i] - lo[i]) * random.uniform(0, 1)
    root_axis = random.randint(0, dim - 1)
    yield [lo[i] if i == root_axis else t_proj(lo, hi, i) for i in range(dim)]


def find_boundaries(r: mdt.Rec, search) -> mdt.Rec:
    diag = r.top - r.bot
    dim = len(r.bot)
    zero_vec = tuple(np.zeros_like(r.bot))
    basis = {tuple(b) for b in mdt.basis_vecs(dim)}
    axis_frame = ((b, r.bot + diag*np.array(b)) for b in 
                  fn.chain(basis, [zero_vec]))
    proj_vecs = list(fn.cat([[ProjVec(r, b2) for b2 in basis - {b}]
                             for b, r in axis_frame]))
    intersects = (projections(r.top, v, [search])[0] for v in proj_vecs)
    intersects = np.array([m for _, m, _ in intersects if m is not None])
    return mdt.Rec(intersects.min(axis=0), intersects.max(axis=1))
