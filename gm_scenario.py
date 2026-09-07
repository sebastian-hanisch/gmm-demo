"""Zufällige 2D-Punktwolken für die GMM-Demo: k Gauß-Cluster auf einem Ring ("blobs", wie
kmeans-demo) oder k nicht-konvexe Halbkreis-Bögen ("moons", generalisiert wie in
dbscan-demo/kmeans-demo) - bei "blobs" bekommt jeder Cluster eine EIGENE Kovarianzmatrix
statt nur einer Standardabweichung, bei "moons" bekommt jeder PUNKT eine an die lokale
Bogen-Tangente ausgerichtete Kovarianz (siehe `_generate_moons`). Zwei unabhängige
Schwierigkeitsachsen zusätzlich zu `spread` (Basis-Überlappung, wie kmeans-demo), bei
BEIDEN Formen wirksam:

- `elongation`: 0 = kreisförmige Streuung, 1 = stark elliptisch. Bei "blobs" zeigt die
  lange Achse (leicht verrauscht) auf den nächsten Cluster auf dem Ring - das ist die
  Achse, an der k-Means' Annahme kugelförmiger Cluster sichtbar bricht. Bei "moons" zeigt
  die lange Achse ENTLANG der Bogen-Tangente an jedem Punkt - ein schmales bis breites
  Band statt eines kreisrunden Streubereichs um die ideale Kurve.
- `variance_imbalance`: 0 = alle Gruppen gleich groß gestreut, 1 = Gruppe 0 deutlich
  diffuser als die übrigen (isotrop, unabhängig von elongation) - die zweite,
  eigenständige k-Means-Annahme (gleiche Varianz über alle Gruppen), die bricht.
"""

from dataclasses import dataclass

import numpy as np

RING_RADIUS = 3.0
ARC_RADIUS = 2.5
ARC_RING_RADIUS = 6.5
MIN_STD_FRACTION = 0.05


@dataclass(frozen=True)
class ClusteringInstance:
    points: tuple  # ((x, y), ...), Erzeugungsreihenfolge nach Cluster gruppiert
    true_labels: tuple
    true_centers: tuple  # ((x, y), ...), k Einträge - bei "moons" der Schwerpunkt je Bogen
    true_covariances: tuple  # (((a, b), (c, d)), ...), k Einträge, 2x2 Kovarianzmatrizen -
    # bei "moons" die Kovarianz am Bogen-Mittelpunkt, repräsentativ (variiert dort real
    # punktweise mit der Tangentenrichtung, siehe `_generate_moons`)
    shape: str  # "blobs" oder "moons"
    k: int

    @property
    def n_points(self):
        return len(self.points)

    def as_array(self):
        return np.array(self.points, dtype=float)


def _cluster_base_stds(k, spread, variance_imbalance):
    """Isotrope Basis-Streuung je Cluster, bevor Elongation angewendet wird. Cluster 0
    wird mit wachsendem variance_imbalance zunehmend diffuser als die übrigen - unabhängig
    davon, ob die Cluster später auch noch elliptisch verzerrt werden. Dieselbe Grund-Idee
    wie dbscan-demo/hdbscan-demos density_imbalance (Gruppe 0 diffuser bei gleicher
    Punktzahl), aber mit bewusst aggressiveren Koeffizienten (4.0/0.5 statt deren 1.0/0.6):
    die Kovarianz-Ellipsen dieser Demo müssen den Unterschied auch bei nur k=2-3 sichtbar
    ausgeprägten Gruppen klar zeigen, nicht erst bei dbscan-typischen 150+ Punkten."""
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


def _generate_blobs(n_points, k, spread, elongation, variance_imbalance, rng):
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
    counts[-1] += n_points - counts.sum()

    points_per_cluster = []
    labels_per_cluster = []
    for i in range(k):
        pts = rng.multivariate_normal(mean=true_centers[i], cov=covariances[i], size=counts[i])
        points_per_cluster.append(pts)
        labels_per_cluster.append(np.full(counts[i], i))

    points = np.concatenate(points_per_cluster, axis=0)
    labels = np.concatenate(labels_per_cluster, axis=0)
    return points, labels, true_centers, covariances


def _sample_along_tangent(x, y, tangent_angle, base_std, elongation, rng):
    """Sampelt Rauschen um (x, y) mit einer an `tangent_angle` ausgerichteten Ellipse
    (lange Achse ENTLANG der Tangente bei elongation=1, kreisförmig bei elongation=0) -
    dieselbe major/minor-Konstruktion wie `_covariance_matrix`, aber punktweise und ohne
    den Umweg über eine explizite Kovarianzmatrix je Punkt."""
    major = base_std * (1.0 + 2.0 * elongation)
    minor = base_std * max(1.0 - 0.7 * elongation, MIN_STD_FRACTION)
    z_major = rng.normal(0.0, major, size=x.shape)
    z_minor = rng.normal(0.0, minor, size=x.shape)
    cos_a, sin_a = np.cos(tangent_angle), np.sin(tangent_angle)
    dx = cos_a * z_major - sin_a * z_minor
    dy = sin_a * z_major + cos_a * z_minor
    return np.stack([x + dx, y + dy], axis=1)


def _generate_moons(n_points, k, spread, elongation, variance_imbalance, rng):
    """k=2: das klassische "two moons"-Beispiel (wie in dbscan-demo/kmeans-demo). k>2: k
    Halbkreis-Bögen wie Blütenblätter auf einem Ring, konkave Seite zum Zentrum.
    variance_imbalance wirkt wie bei "blobs" (Gruppe 0 diffuser). elongation richtet die
    lange Achse an der lokalen Bogen-TANGENTE aus statt an einem festen Nachbar-Cluster -
    bei elongation=1 entsteht ein schmales, lang gezogenes Band statt eines breiten
    kreisrunden Streubereichs um die ideale Kurve."""
    counts = np.full(k, n_points // k)
    counts[-1] += n_points - counts.sum()
    base_stds = _cluster_base_stds(k, spread, variance_imbalance) * 0.3 * (ARC_RADIUS / RING_RADIUS)

    if k == 2:
        t1 = rng.uniform(0, np.pi, counts[0])
        x1 = ARC_RADIUS * np.cos(t1)
        y1 = ARC_RADIUS * np.sin(t1)
        tangent1 = t1 + np.pi / 2

        t2 = rng.uniform(0, np.pi, counts[1])
        x2 = ARC_RADIUS * (1 - np.cos(t2))
        y2 = ARC_RADIUS * (0.5 - np.sin(t2))
        tangent2 = t2 - np.pi / 2

        pts1 = _sample_along_tangent(x1, y1, tangent1, base_stds[0], elongation, rng)
        pts2 = _sample_along_tangent(x2, y2, tangent2, base_stds[1], elongation, rng)
        points = np.concatenate([pts1, pts2], axis=0)
        labels = np.concatenate([np.zeros(counts[0], dtype=int), np.ones(counts[1], dtype=int)])
        true_centers = np.stack([points[labels == i].mean(axis=0) for i in range(k)])
        covariances = np.array([
            _covariance_matrix(base_stds[i], elongation, [np.pi / 2, -np.pi / 2][i]) for i in range(k)
        ])
        return points, labels, true_centers, covariances

    layout_angles = np.linspace(0, 2 * np.pi, k, endpoint=False) + rng.uniform(-0.1, 0.1, size=k)
    arc_centers = np.stack(
        [ARC_RING_RADIUS * np.cos(layout_angles), ARC_RING_RADIUS * np.sin(layout_angles)], axis=1
    )

    points_per_group, labels_per_group = [], []
    for i in range(k):
        t = rng.uniform(0, np.pi, counts[i])
        local_x = ARC_RADIUS * np.cos(t)
        local_y = ARC_RADIUS * np.sin(t)
        # Um layout_angle_i + pi rotieren, damit die konkave Seite des Bogens zum
        # Ringzentrum zeigt (Blütenblatt-Anordnung), statt nach außen - die Tangente
        # rotiert exakt mit.
        rot = layout_angles[i] + np.pi
        cos_r, sin_r = np.cos(rot), np.sin(rot)
        rx = cos_r * local_x - sin_r * local_y
        ry = sin_r * local_x + cos_r * local_y
        tangent = t + np.pi / 2 + rot
        pts = _sample_along_tangent(rx + arc_centers[i, 0], ry + arc_centers[i, 1], tangent, base_stds[i], elongation, rng)
        points_per_group.append(pts)
        labels_per_group.append(np.full(counts[i], i))

    points = np.concatenate(points_per_group, axis=0)
    labels = np.concatenate(labels_per_group, axis=0)
    true_centers = np.stack([points[labels == i].mean(axis=0) for i in range(k)])
    covariances = np.array([
        _covariance_matrix(base_stds[i], elongation, layout_angles[i] + np.pi + np.pi / 2) for i in range(k)
    ])
    return points, labels, true_centers, covariances


def generate_instance(n_points, k, spread, elongation, variance_imbalance, seed, shape="blobs"):
    """spread steuert die Basis-Überlappung (wie kmeans-demo), elongation und
    variance_imbalance sind zwei unabhängige zusätzliche Schwierigkeitsachsen, bei BEIDEN
    Formen wirksam (siehe Modul-Docstring)."""
    rng = np.random.default_rng(seed)

    if shape == "moons":
        points, labels, true_centers, covariances = _generate_moons(
            n_points, k, spread, elongation, variance_imbalance, rng
        )
    else:
        points, labels, true_centers, covariances = _generate_blobs(
            n_points, k, spread, elongation, variance_imbalance, rng
        )

    return ClusteringInstance(
        points=tuple(map(tuple, points.tolist())),
        true_labels=tuple(int(l) for l in labels),
        true_centers=tuple(map(tuple, true_centers.tolist())),
        true_covariances=tuple(tuple(map(tuple, cov.tolist())) for cov in covariances),
        shape=shape,
        k=k,
    )
