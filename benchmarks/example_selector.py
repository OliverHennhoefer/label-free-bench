"""Interface example: rank detectors by their agreement with the other candidates.

This reproduces the library's model-centrality baseline (MC) to show the selector
contract. It is not a research hypothesis.
"""

import labelfree as lf


def select(batch):
    # batch.X: selection normals; batch.scores: candidate x row, higher = more anomalous.
    # Return one value per candidate, larger preferred, or None to abstain.
    return lf.model_centrality_scores(batch.scores)
