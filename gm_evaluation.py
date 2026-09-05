"""Rand-Index (from scratch) sowie der Kovarianz-Typ-Vergleich bei festem k - das
agglomerative-demo-Linkage-Vergleich-Äquivalent für die "Was bringt volle Kovarianz?"-
Sektion."""

import numpy as np

from gm_algorithm import run


def rand_index(true_labels, pred_labels):
    """Anteil der Punktpaare, bei denen beide Partitionen übereinstimmen (entweder beide
    im selben Cluster oder beide in unterschiedlichen)."""
    true_arr = np.asarray(true_labels)
    pred_arr = np.asarray(pred_labels)
    n = len(true_arr)
    if n < 2:
        return 1.0

    iu = np.triu_indices(n, k=1)
    same_true = (true_arr[:, None] == true_arr[None, :])[iu]
    same_pred = (pred_arr[:, None] == pred_arr[None, :])[iu]
    return float(np.mean(same_true == same_pred))


def covariance_type_comparison(data, true_labels, k, seed):
    """Rand-Index je Kovarianz-Typ, alle beim GLEICHEN k und Seed bei Konvergenz
    berechnet - macht direkt vergleichbar, wie stark das Ergebnis von der
    Kovarianz-Annahme abhängt."""
    from gm_constants import COVARIANCE_TYPES

    scores = {}
    for covariance_type in COVARIANCE_TYPES:
        result = run(data, k, covariance_type, seed)
        labels = result.hard_labels()
        scores[covariance_type] = rand_index(true_labels, labels)
    return scores
