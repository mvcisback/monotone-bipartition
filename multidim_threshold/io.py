from pathlib import Path

import capnp

from multidim_threshold import utils

RecSetCapn = capnp.load(str(Path(__file__).parent / "rec_set.capnp"))
RecSet = RecSetCapn.RecSet
Rec = RecSetCapn.Rec
RecSetCollection = RecSetCapn.RecSetCollection

lmap = lambda f, xs: list(map(f, xs))

############ Load ##############

def load_rec(rec):
    return utils.Rec(tuple(rec.bot), tuple(rec.top))


def load_rec_set(rec_set):
    return utils.Result(unexplored=lmap(load_rec, rec_set.recs))


def load(fp):
    return map(load_rec_set, RecSetCollection.read(fp).recSets)


########## Write ###############

def to_capnp_rec(rec):
    return Rec.new_message(bot=lmap(float, rec.bot), top=lmap(float, rec.top))


def to_capnp_rec_set(res):
    return RecSet.new_message(recs=lmap(to_capnp_rec, res.unexplored))


def write(results, fp):
    _rec_sets = lmap(to_capnp_rec_set, results)
    RecSetCollection.new_message(recSets=_rec_sets).write(fp)
