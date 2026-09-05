"""Zufällige 2D-Punktwolken für die GMM-Demo: k Gauß-Cluster auf einem Ring
(wie kmeans-demo), aber jeder Cluster bekommt eine EIGENE Kovarianzmatrix statt nur
einer eigenen Standardabweichung. Zwei unabhängige Schwierigkeitsachsen zusätzlich zu
`spread` (Basis-Überlappung, wie kmeans-demo):

- `elongation`: 0 = kreisförmige Cluster, 1 = stark elliptisch, mit der langen Achse
  jeweils (leicht verrauscht) auf den nächsten Cluster auf dem Ring ausgerichtet - das ist
  die Achse, an der k-Means' Annahme kugelförmiger Cluster sichtbar bricht.
- `variance_imbalance`: 0 = alle Cluster gleich groß gestreut, 1 = Cluster 0 deutlich
  diffuser als die übrigen (isotrop, unabhängig von elongation) - die zweite,
  eigenständige k-Means-Annahme (gleiche Varianz über alle Cluster), die bricht.
"""

from dataclasses import dataclass

import numpy as np

RING_RADIUS = 3.0
MIN_STD_FRACTION = 0.05


@dataclass(frozen=True)
class ClusteringInstance:
    points: tuple  # ((x, y), ...), Erzeugungsreihenfolge nach Cluster gruppiert
    true_labels: tuple
    true_centers: tuple  # ((x, y), ...), k Einträge
    true_covariances: tuple  # (((a, b), (c, d)), ...), k Einträge, 2x2 Kovarianzmatrizen
    k: int

    @property
    def n_points(self):
        return len(self.points)

    def as_array(self):
        return np.array(self.points, dtype=float)


def _cluster_base_stds(k, spread, variance_imbalance):
    """Isotrope Basis-Streuung je Cluster, bevor Elongation angewendet wird. Cluster 0
    wird mit wachsendem variance_imbalance zunehmend diffuser als die übrigen - unabhängig
    davon, ob die Cluster später auch noch elliptisch verzerrt werden."""
    base_std = max(spread, MIN_STD_FRACTION) * RING_RADIUS
    stds = np.full(k, base_std)
    if k > 1:
        stds[0] *= 1 + 4.0 * variance_imbalance
        stds[1:] *= max(1 - 0.5 * variance_imbalance, MIN_STD_FRACTION)
    return stds


def _covariance_matrix(base_std, elongation, angle):
    """Baut eine 2x2-Kovarianzmatrix aus einer Basis-Streuung, einem Elongations-Grad
    (0 = kreisförmig, 1 = stark elliptisch) und einem Rotationswinkel. minor/major sind
    so gewählt, dass elongation=0 exakt die isotrope Basis-Streuung reproduziert."""
    major = base_std * (1.0 + 2.0 * elongation)
    minor = base_std * max(1.0 - 0.7 * elongation, MIN_STD_FRACTION)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    diag = np.diag([major ** 2, minor ** 2])
    return rotation @ diag @ rotation.T


def generate_instance(n_points, k, spread, elongation, variance_imbalance, seed):
    """spread steuert die Basis-Überlappung (wie kmeans-demo), elongation und
    variance_imbalance sind zwei unabhängige zusätzliche Schwierigkeitsachsen (siehe
    Modul-Docstring)."""
    rng = np.random.default_rng(seed)

    angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.15, 0.15, size=k)
    true_centers = np.stack([RING_RADIUS * np.cos(angles), RING_RADIUS * np.sin(angles)], axis=1)

    base_stds = _cluster_base_stds(k, spread, variance_imbalance)
    # Die Ellipsen-Rotation zeigt (mit etwas Jitter) bewusst auf den JEWEILS NÄCHSTEN
    # Cluster auf dem Ring, statt komplett zufällig zu sein - nur so reicht die lange
    # Achse gezielt in Nachbar-Territorium hinein und macht kugelförmige (isotrope)
    # Kovarianz zuverlässig scheitern, statt nur bei zufälligem Rotations-Glück.
    neighbor_vectors = np.roll(true_centers, -1, axis=0) - true_centers
    rotation_angles = np.arctan2(neighbor_vectors[:, 1], neighbor_vectors[:, 0])
    rotation_angles = rotation_angles + rng.uniform(-0.2, 0.2, size=k)
    covariances = np.array([
        _covariance_matrix(base_stds[i], elongation, rotation_angles[i]) for i in range(k)
    ])

    counts = np.full(k, n_points // k)
    counts[: n_points % k] += 1

    points_per_cluster = []
    labels_per_cluster = []
    for i in range(k):
        pts = rng.multivariate_normal(mean=true_centers[i], cov=covariances[i], size=counts[i])
        points_per_cluster.append(pts)
        labels_per_cluster.append(np.full(counts[i], i))

    points = np.concatenate(points_per_cluster, axis=0)
    labels = np.concatenate(labels_per_cluster, axis=0)

    return ClusteringInstance(
        points=tuple(map(tuple, points.tolist())),
        true_labels=tuple(int(l) for l in labels),
        true_centers=tuple(map(tuple, true_centers.tolist())),
        true_covariances=tuple(tuple(map(tuple, cov.tolist())) for cov in covariances),
        k=k,
    )
