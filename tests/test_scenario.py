import numpy as np
import pytest

from gm_scenario import generate_instance


def test_reproducible_with_same_seed():
    a = generate_instance(100, 3, 0.3, 0.5, 0.4, seed=42)
    b = generate_instance(100, 3, 0.3, 0.5, 0.4, seed=42)
    assert a.points == b.points
    assert a.true_labels == b.true_labels


def test_different_seed_gives_different_points():
    a = generate_instance(100, 3, 0.3, 0.5, 0.4, seed=1)
    b = generate_instance(100, 3, 0.3, 0.5, 0.4, seed=2)
    assert a.points != b.points


def test_n_points_and_k_respected():
    instance = generate_instance(97, 4, 0.3, 0.0, 0.0, seed=5)
    assert instance.n_points == 97
    assert instance.k == 4
    assert set(instance.true_labels) == {0, 1, 2, 3}


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_elongation_increases_empirical_eigenvalue_ratio(seed):
    """Höhere Elongation muss das Achsenverhältnis (größter/kleinster Eigenwert) der
    empirischen Kovarianz jedes Clusters nachweislich vergrößern - das ist die
    geometrische Eigenschaft, die GMM mit voller Kovarianz ausnutzt und k-Means nicht."""
    low = generate_instance(400, 1, 0.3, elongation=0.0, variance_imbalance=0.0, seed=seed)
    high = generate_instance(400, 1, 0.3, elongation=0.9, variance_imbalance=0.0, seed=seed)

    def eigenvalue_ratio(instance):
        points = np.array(instance.points)
        cov = np.cov(points, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
        return eigenvalues.max() / eigenvalues.min()

    assert eigenvalue_ratio(high) > eigenvalue_ratio(low) * 2


def test_variance_imbalance_makes_cluster_zero_more_diffuse_independent_of_elongation():
    """variance_imbalance muss Cluster 0 diffuser als die uebrigen machen, unabhaengig
    davon ob elongation gleichzeitig aktiv ist."""
    for elongation in (0.0, 0.7):
        instance = generate_instance(
            600, 3, 0.2, elongation=elongation, variance_imbalance=0.9, seed=9
        )
        points = np.array(instance.points)
        labels = np.array(instance.true_labels)
        variances = [np.trace(np.cov(points[labels == i], rowvar=False)) for i in range(3)]
        assert variances[0] > 2 * max(variances[1], variances[2])


def test_default_shape_is_blobs():
    instance = generate_instance(90, 3, 0.25, 0.0, 0.0, seed=1)
    assert instance.shape == "blobs"


def test_moons_shape_produces_two_balanced_groups():
    instance = generate_instance(100, 2, 0.1, 0.0, 0.0, seed=5, shape="moons")
    assert instance.shape == "moons"
    labels = np.array(instance.true_labels)
    counts = np.bincount(labels, minlength=2)
    assert counts[0] == 50 and counts[1] == 50


def test_moons_shape_with_k_greater_than_two_produces_k_balanced_arcs():
    instance = generate_instance(200, 4, 0.1, 0.0, 0.0, seed=6, shape="moons")
    labels = np.array(instance.true_labels)
    counts = np.bincount(labels, minlength=4)
    assert counts.min() == counts.max() == 50


def test_variance_imbalance_makes_group_zero_more_diffuse_for_moons_too():
    """Fuer "moons" dominiert die Bogenlaenge selbst die rohe Punktkovarianz (anders als
    bei "blobs", wo die Kovarianz ausschliesslich vom Rauschen kommt) - deshalb wird hier
    der Abstand jedes Punkts zur eigenen idealen Bogenkurve gemessen (das reine
    Rausch-Signal), nicht die rohe Kovarianz-Spur."""
    import gm_scenario as S

    instance = generate_instance(600, 2, 0.1, elongation=0.0, variance_imbalance=0.9, seed=9, shape="moons")
    points = np.array(instance.points)
    labels = np.array(instance.true_labels)

    t_dense = np.linspace(0, np.pi, 2000)
    curve0 = np.stack([S.ARC_RADIUS * np.cos(t_dense), S.ARC_RADIUS * np.sin(t_dense)], axis=1)
    curve1 = np.stack([S.ARC_RADIUS * (1 - np.cos(t_dense)), S.ARC_RADIUS * (0.5 - np.sin(t_dense))], axis=1)

    def residual_std(group_points, curve):
        dists = np.array([np.sqrt(((curve - p) ** 2).sum(axis=1)).min() for p in group_points])
        return dists.std()

    std0 = residual_std(points[labels == 0], curve0)
    std1 = residual_std(points[labels == 1], curve1)
    assert std0 > std1 * 1.5


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_moons_elongation_stretches_noise_along_the_arc_tangent(seed):
    """Bei "moons" muss hohe elongation das Rauschen ENTLANG der Bogen-Tangente
    strecken statt quer dazu - geprueft ueber die Projektion jedes Punkt-Residuums
    (Punkt minus naechster Punkt auf der idealen Kurve) auf Tangential- vs.
    Normalenrichtung. Das ist das moons-Gegenstueck zu
    test_elongation_increases_empirical_eigenvalue_ratio fuer blobs."""
    import gm_scenario as S

    def tangential_normal_ratio(elongation):
        instance = S.generate_instance(300, 2, 0.1, elongation, 0.0, seed=seed, shape="moons")
        points = np.array(instance.points)
        labels = np.array(instance.true_labels)
        group0 = points[labels == 0]
        # ideale Kurve fuer Gruppe 0 (t1 in [0, pi], siehe _generate_moons): dicht
        # abgetastet, dann pro Punkt den naechsten Kurvenpunkt und dessen Tangente finden.
        t_dense = np.linspace(0, np.pi, 2000)
        curve = np.stack([S.ARC_RADIUS * np.cos(t_dense), S.ARC_RADIUS * np.sin(t_dense)], axis=1)
        nearest_idx = np.array([np.argmin(((curve - p) ** 2).sum(axis=1)) for p in group0])
        nearest_t = t_dense[nearest_idx]
        tangent = np.stack([-np.sin(nearest_t), np.cos(nearest_t)], axis=1)
        normal = np.stack([np.cos(nearest_t), np.sin(nearest_t)], axis=1)
        residual = group0 - curve[nearest_idx]
        tangential = (residual * tangent).sum(axis=1)
        normal_comp = (residual * normal).sum(axis=1)
        return tangential.std() / normal_comp.std()

    low_ratio = tangential_normal_ratio(0.0)
    high_ratio = tangential_normal_ratio(1.0)
    assert high_ratio > low_ratio * 1.5
