"""Cross-family tail consensus.

Hypothesis: on normal-only selection rows, the anomaly-relevant structure is the
periphery. A candidate whose top-5% rows match the periphery agreed by the other
detector families (one vote per family, own family excluded) tracks low density
better. Rank each candidate by the overlap of its top-5% set with the top 5% of
the mean-rank consensus of the other families, using fractional membership at
ties. No refits. Abstains when fewer than two families are present.
"""

import numpy as np
from scipy.stats import rankdata

TAIL = 0.05


def _membership(values, k):
    """Fractional top-k membership: ties at the boundary share the remainder."""
    n = values.shape[0]
    membership = np.zeros(n)
    order = np.argsort(-values, kind="stable")
    sorted_values = values[order]
    filled = 0.0
    start = 0
    while start < n and filled < k:
        stop = start
        while stop + 1 < n and sorted_values[stop + 1] == sorted_values[start]:
            stop += 1
        group = order[start : stop + 1]
        take = min(float(len(group)), k - filled)
        membership[group] = take / len(group)
        filled += take
        start = stop + 1
    return membership


def select(batch):
    scores = np.asarray(batch.scores, dtype=float)
    families = list(batch.families)
    n_candidates, n_rows = scores.shape
    unique = sorted(set(families))
    if len(unique) < 2 or n_rows < 2 or not np.all(np.isfinite(scores)):
        return None
    ranks = np.vstack([rankdata(row) for row in scores])
    family_rank = {
        family: ranks[[i for i, f in enumerate(families) if f == family]].mean(axis=0)
        for family in unique
    }
    k = max(1.0, TAIL * n_rows)
    values = []
    for i in range(n_candidates):
        others = [family_rank[f] for f in unique if f != families[i]]
        consensus = np.mean(others, axis=0)
        own = _membership(scores[i], k)
        agreed = _membership(consensus, k)
        values.append(float(np.dot(own, agreed) / k))
    return values
