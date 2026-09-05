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
