import pytest

from gm_evaluation import covariance_type_comparison, rand_index
from gm_scenario import generate_instance


def test_rand_index_is_one_for_identical_partitions():
    assert rand_index([0, 0, 1, 1], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_rand_index_is_one_up_to_relabeling():
    assert rand_index([0, 0, 1, 1], [5, 5, 9, 9]) == pytest.approx(1.0)


def test_rand_index_penalizes_disagreement():
    score = rand_index([0, 0, 1, 1], [0, 1, 0, 1])
    assert score < 0.6


@pytest.mark.parametrize("seed", [3, 8, 9])
def test_full_covariance_beats_spherical_on_elongated_rotated_clusters(seed):
    """Kern-Nachweis der Demo: bei stark elliptischen Gruppen, deren lange Achse gezielt
    auf den jeweils nächsten Cluster ausgerichtet ist, scheitert die kugelförmige
    (isotrope) Kovarianz-Annahme - k-Means' implizite Annahme - während volle Kovarianz
    die wahre Gruppenstruktur sauber wiederfindet. Seeds per Sweep-Skript verifiziert
    (siehe project_gmm_demo_venv.md)."""
    instance = generate_instance(200, 3, 0.25, elongation=0.9, variance_imbalance=0.0, seed=seed)
    scores = covariance_type_comparison(instance.as_array(), instance.true_labels, 3, seed=1)

    assert scores["full"] > 0.92
    assert scores["spherical"] < 0.72
    assert scores["full"] - scores["spherical"] > 0.2


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_tied_covariance_fails_on_unequal_variance_where_others_succeed(seed):
    """Zweiter, unabhängiger Nachweis: bei stark unterschiedlich gestreuten (aber
    kreisförmigen) Gruppen scheitert die GEBUNDENE Kovarianz-Annahme (eine gemeinsame
    Form für alle Cluster erzwingt implizit gleiche Varianz überall), während Typen mit
    eigener Varianz je Komponente (spherical/diag/full) das Größen-Ungleichgewicht
    problemlos abbilden. Seeds per Sweep-Skript verifiziert."""
    instance = generate_instance(200, 3, 0.25, elongation=0.0, variance_imbalance=0.9, seed=seed)
    scores = covariance_type_comparison(instance.as_array(), instance.true_labels, 3, seed=1)

    assert scores["tied"] < 0.75
    assert scores["spherical"] > 0.92
    assert scores["full"] > 0.92


def test_all_covariance_types_agree_on_simple_scenario():
    instance = generate_instance(120, 3, 0.2, elongation=0.0, variance_imbalance=0.0, seed=1)
    scores = covariance_type_comparison(instance.as_array(), instance.true_labels, 3, seed=1)
    assert min(scores.values()) > 0.9
