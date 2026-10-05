"""Orakel-Test: Schritt-für-Schritt-Vergleich des EM-Protokolls mit sklearn.mixture.GaussianMixture.

Schritt j des Protokolls (M-Schritt dann E-Schritt, ausgehend von identischen Startparametern) ist
genau der sklearn-Zustand nach `max_iter=j` mit `tol=0`. Verglichen werden Gewichte, Mittelwerte,
Kovarianzen, Responsibilities und Log-Likelihood auf 1e-6, nicht nur der Fixpunkt. Zusätzlich:
Log-Likelihood gegen scipy.stats.multivariate_normal, Rand-Index gegen sklearn.metrics.rand_score,
Monotonie über viele Zufallsinstanzen."""

import warnings

import numpy as np
import pytest

from gm_algorithm import init_params, run_from_init
from gm_constants import COVARIANCE_TYPES, REG_COVAR
from gm_evaluation import rand_index
from gm_scenario import generate_instance

mixture = pytest.importorskip("sklearn.mixture")
metrics = pytest.importorskip("sklearn.metrics")
stats = pytest.importorskip("scipy.stats")


def _project(cov, covariance_type):
    if covariance_type in ("full", "tied"):
        return cov.copy()
    if covariance_type == "diag":
        return np.diag(np.diag(cov))
    return np.eye(2) * (np.trace(cov) / 2)


def _precisions(covs, covariance_type):
    if covariance_type == "full":
        return np.array([np.linalg.inv(c) for c in covs])
    if covariance_type == "tied":
        return np.linalg.inv(covs[0])
    if covariance_type == "diag":
        return np.array([1.0 / np.diag(c) for c in covs])
    return np.array([1.0 / c[0, 0] for c in covs])


def _sk_covariances(model, covariance_type, k):
    c = model.covariances_
    if covariance_type == "full":
        return c
    if covariance_type == "tied":
        return np.array([c] * k)
    if covariance_type == "diag":
        return np.array([np.diag(x) for x in c])
    return np.array([np.eye(2) * x for x in c])


def test_em_trajectory_matches_sklearn_stepwise():
    rng = np.random.default_rng(1)
    for trial in range(24):
        covariance_type = COVARIANCE_TYPES[trial % 4]
        k = int(rng.integers(2, 5))
        shape = "blobs" if trial % 3 else "moons"
        instance = generate_instance(
            int(rng.integers(30, 100)), k, float(rng.uniform(0.1, 0.9)), float(rng.uniform(0, 1)),
            float(rng.uniform(0, 1)), int(rng.integers(0, 10**6)), shape=shape,
        )
        data = instance.as_array()
        w0, m0, c0 = init_params(data, k, np.random.default_rng(int(rng.integers(0, 10**6))))
        c0 = np.array([_project(c, covariance_type) for c in c0])
        result = run_from_init(data, w0, m0, c0, covariance_type, max_iter=8, tol=0.0)

        log_likelihoods = [s.log_likelihood for s in result.steps]
        assert all(b >= a - 1e-6 for a, b in zip(log_likelihoods, log_likelihoods[1:]))

        for j in (1, len(result.steps) - 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = mixture.GaussianMixture(
                    k, covariance_type=covariance_type, weights_init=w0, means_init=m0,
                    precisions_init=_precisions(c0, covariance_type), max_iter=j, tol=0.0,
                    reg_covar=REG_COVAR, n_init=1,
                ).fit(data)
            step = result.steps[j]
            np.testing.assert_allclose(step.weights, model.weights_, atol=1e-6)
            np.testing.assert_allclose(step.means, model.means_, atol=1e-6)
            np.testing.assert_allclose(step.covariances, _sk_covariances(model, covariance_type, k), atol=1e-6)
            np.testing.assert_allclose(step.responsibilities, model.predict_proba(data), atol=1e-6)
            assert step.log_likelihood / len(data) == pytest.approx(model.score(data), abs=1e-8)

        # Log-Likelihood des letzten Schritts gegen scipy
        last = result.final_step
        log_comp = np.stack(
            [
                np.log(last.weights[j])
                + stats.multivariate_normal(last.means[j], np.array(last.covariances[j])).logpdf(data)
                for j in range(k)
            ],
            axis=1,
        )
        assert last.log_likelihood == pytest.approx(np.log(np.exp(log_comp).sum(axis=1)).sum(), rel=1e-9)

        labels = result.hard_labels()
        assert rand_index(instance.true_labels, labels) == pytest.approx(
            metrics.rand_score(instance.true_labels, labels), abs=1e-12
        )
