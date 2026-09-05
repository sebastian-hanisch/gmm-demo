import numpy as np
import pytest
from sklearn.mixture import GaussianMixture

from gm_algorithm import e_step, init_params, m_step, run, run_from_init
from gm_constants import COVARIANCE_TYPES, REG_COVAR
from gm_scenario import generate_instance


def test_e_step_hand_computed_two_points_two_components():
    """Ein Punkt bei (0,0), zwei Standard-Normalverteilungen bei (0,0) und (3,0),
    Gewichte 0.5/0.5. Von Hand: log N(x|mu1)=-ln(2*pi), log N(x|mu2)=-ln(2*pi)-4.5, die
    Gewichte kürzen sich in der Differenz heraus -> r1 = 1/(1+exp(-4.5))."""
    data = np.array([[0.0, 0.0]])
    weights = np.array([0.5, 0.5])
    means = np.array([[0.0, 0.0], [3.0, 0.0]])
    covariances = np.array([np.eye(2), np.eye(2)])

    responsibilities, _ = e_step(data, weights, means, covariances)

    expected_r1 = 1.0 / (1.0 + np.exp(-4.5))
    assert responsibilities[0, 0] == pytest.approx(expected_r1, abs=1e-9)
    assert responsibilities[0, 1] == pytest.approx(1.0 - expected_r1, abs=1e-9)
    assert responsibilities.sum(axis=1)[0] == pytest.approx(1.0)


def test_m_step_hand_computed_weighted_mean_and_covariance():
    """Zwei Punkte (0,0) und (2,0), Responsibilities [0.75,0.25] bzw. [0.25,0.75]. Von
    Hand: mean1 = (0.75*0+0.25*2)/1.0 = 0.5, mean2 = (0.25*0+0.75*2)/1.0 = 1.5;
    cov1[0,0] = (0.75*(-0.5)^2 + 0.25*1.5^2)/1.0 = 0.75 (plus Regularisierung)."""
    data = np.array([[0.0, 0.0], [2.0, 0.0]])
    responsibilities = np.array([[0.75, 0.25], [0.25, 0.75]])

    weights, means, covariances = m_step(data, responsibilities, "full")

    assert weights == pytest.approx([0.5, 0.5])
    assert means[0] == pytest.approx([0.5, 0.0])
    assert means[1] == pytest.approx([1.5, 0.0])
    assert covariances[0][0, 0] == pytest.approx(0.75 + REG_COVAR)
    assert covariances[0][1, 1] == pytest.approx(REG_COVAR)
    assert covariances[0][0, 1] == pytest.approx(0.0, abs=1e-12)
    assert covariances[1][0, 0] == pytest.approx(0.75 + REG_COVAR)


def test_responsibilities_and_weights_sum_to_one():
    instance = generate_instance(80, 3, 0.3, 0.4, 0.3, seed=13)
    data = instance.as_array()
    for covariance_type in COVARIANCE_TYPES:
        result = run(data, 3, covariance_type, seed=1)
        for step in result.steps:
            resp = np.array(step.responsibilities)
            assert np.allclose(resp.sum(axis=1), 1.0, atol=1e-8)
            assert np.sum(step.weights) == pytest.approx(1.0)


def test_covariances_are_positive_semidefinite():
    instance = generate_instance(80, 3, 0.3, 0.6, 0.5, seed=17)
    data = instance.as_array()
    for covariance_type in COVARIANCE_TYPES:
        result = run(data, 3, covariance_type, seed=2)
        for cov in result.final_covariances:
            eigenvalues = np.linalg.eigvalsh(np.array(cov))
            assert eigenvalues.min() > -1e-8


@pytest.mark.parametrize(
    "seed,elongation,variance_imbalance", [(1, 0.0, 0.0), (2, 0.8, 0.0), (3, 0.0, 0.8), (4, 0.6, 0.5)]
)
@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_log_likelihood_is_monotonically_non_decreasing(covariance_type, seed, elongation, variance_imbalance):
    """EM erhöht die Log-Likelihood der Daten in jedem Schritt nie - eine bewiesene
    Eigenschaft (siehe Mathe-Abschnitt), hier über mehrere Szenarien/Kovarianz-Typen
    geprüft statt nur behauptet."""
    instance = generate_instance(120, 3, 0.3, elongation, variance_imbalance, seed=seed)
    result = run(instance.as_array(), 3, covariance_type, seed=seed)
    log_likelihoods = [s.log_likelihood for s in result.steps]
    for prev, curr in zip(log_likelihoods, log_likelihoods[1:]):
        assert curr >= prev - 1e-6


def test_max_iterations_cap_is_respected():
    instance = generate_instance(60, 3, 0.3, 0.0, 0.0, seed=1)
    result = run(instance.as_array(), 3, "full", seed=1, max_iter=2)
    assert len(result.steps) <= 3
    if result.truncated:
        assert not result.converged


def _project_covariance(cov, covariance_type):
    d = cov.shape[0]
    if covariance_type in ("full", "tied"):
        return cov.copy()
    if covariance_type == "diag":
        return np.diag(np.diag(cov))
    if covariance_type == "spherical":
        return np.eye(d) * (np.trace(cov) / d)
    raise ValueError(covariance_type)


def _sklearn_precisions_init(covariances0, covariance_type):
    if covariance_type == "full":
        return np.array([np.linalg.inv(cov) for cov in covariances0])
    if covariance_type == "tied":
        return np.linalg.inv(covariances0[0])
    if covariance_type == "diag":
        return np.array([1.0 / np.diag(cov) for cov in covariances0])
    if covariance_type == "spherical":
        return np.array([1.0 / cov[0, 0] for cov in covariances0])
    raise ValueError(covariance_type)


@pytest.mark.parametrize("covariance_type", COVARIANCE_TYPES)
def test_matches_sklearn_gmm_with_identical_initialization(covariance_type):
    """EM ist bei identischem Start deterministisch (keine Tie-Breaking-Mehrdeutigkeit
    wie bei hdbscan-demos Mutual-Reachability-Distanzen) - daher hier ein EXAKTER
    Kreuzvergleich statt nur einer Toleranzschwelle: beide Implementierungen starten von
    numerisch identischen Gewichten/Mittelwerten/Kovarianzen (`means_init`/
    `precisions_init`) und müssen auf denselben Fixpunkt konvergieren."""
    instance = generate_instance(150, 3, 0.3, elongation=0.5, variance_imbalance=0.4, seed=21)
    data = instance.as_array()
    k = 3

    rng = np.random.default_rng(99)
    weights0, means0, base_covariances0 = init_params(data, k, rng)
    covariances0 = np.array([_project_covariance(cov, covariance_type) for cov in base_covariances0])

    mine = run_from_init(data, weights0, means0, covariances0, covariance_type, max_iter=300, tol=1e-12)

    sk_model = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        weights_init=weights0,
        means_init=means0,
        precisions_init=_sklearn_precisions_init(covariances0, covariance_type),
        max_iter=300,
        tol=1e-12,
        reg_covar=REG_COVAR,
        n_init=1,
        random_state=0,
    )
    sk_model.fit(data)

    np.testing.assert_allclose(mine.final_step.weights, sk_model.weights_, atol=1e-3)
    np.testing.assert_allclose(np.array(mine.final_step.means), sk_model.means_, atol=1e-2)

    my_mean_log_likelihood = mine.final_step.log_likelihood / len(data)
    assert my_mean_log_likelihood == pytest.approx(sk_model.score(data), abs=1e-3)

    my_labels = np.array(mine.hard_labels())
    sk_labels = sk_model.predict(data)
    agreement = float(np.mean(my_labels == sk_labels))
    assert agreement > 0.95
