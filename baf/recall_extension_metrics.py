"""Batch score ensembles and earlier-only F2 member selection."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .recall_weighted_metrics import aligned, check_data, earlier_months, learn, measure


def mean_scores(items):
    if not items:
        raise ValueError("An ensemble requires members")
    for data in items:
        aligned(items[0], data)
    return dict(row_ids=items[0]["row_ids"], labels=items[0]["labels"],
                scores=np.mean([a["scores"] for a in items], axis=0))


def percentile_scores(scores):
    scores = np.asarray(scores)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Invalid percentile population")
    return (rankdata(scores, method="average") - .5) / len(scores)


def rank_blend(anchor, other, weight):
    aligned(anchor, other)
    if not 0 <= weight <= 1:
        raise ValueError("Invalid anchor weight")
    # Only scores, never labels, determine the normalization.
    scores = weight * percentile_scores(anchor["scores"]) + (1-weight) * percentile_scores(other["scores"])
    return dict(row_ids=anchor["row_ids"], labels=anchor["labels"], scores=scores)


def choose(history, names, evaluation):
    values = []
    for name in names:
        past = history[name]
        months = earlier_months(list(past), evaluation)
        policy = learn(past, "pooled", evaluation)
        value = np.mean([measure(past[m], policy)["f2"] for m in months])
        values.append((-value, name))
    if not values:
        raise ValueError("No eligible ensemble member")
    return min(values)[1]


def construct(history, config, evaluation):
    """Return each strategy's current data, eligible history and recipe."""
    months = config["validation_months"]
    earlier = [m for m in months if m < evaluation]
    earlier_months(earlier, evaluation)
    result = {name:(items[evaluation], {m:items[m] for m in earlier}, dict(kind="base",id=name))
              for name,items in history.items()}
    pools = {family:[s["id"] for s in config["candidates"]+config["seed_means"] if s["family"]==family]
             for family in ["window","lightgbm"]}
    past = {name:{m:items[m] for m in earlier} for name,items in history.items()}
    members = {family:choose(past,names,evaluation) for family,names in pools.items()}
    for spec in config["rank_ensembles"]:
        member = members[spec["family"]]
        recipe = dict(kind="rank_blend", anchor=spec["anchor"], member=member, weight=spec["weight"])
        items = {m:rank_blend(history[recipe["anchor"]][m],history[member][m],recipe["weight"]) for m in earlier+[evaluation]}
        result[spec["id"]]=(items[evaluation],{m:items[m] for m in earlier},recipe)
    return result


def leaves(name, config, recipes):
    means={s["id"]:s for s in config["seed_means"]}
    recipe=recipes.get(name,{"kind":"base","id":name})
    if name in means:
        return set(means[name]["members"])
    if recipe["kind"]=="rank_blend":
        return leaves(recipe["anchor"],config,recipes)|leaves(recipe["member"],config,recipes)
    return {name}
