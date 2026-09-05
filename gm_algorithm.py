"""EM-Algorithmus für Gaussian Mixture Models from scratch, mit vollständigem
Iterations-Protokoll (ein Schritt = ein E- oder ein E-nach-M-Zyklus), damit die App
Schritt für Schritt durchblättern kann - analog zu Lloyd's Algorithmus in
kmeans-demo/km_algorithm.py.

Bewusst ohne sklearn zur Laufzeit implementiert. sklearn.mixture.GaussianMixture dient in
tests/ nur als unabhängiger Kreuzvergleich (mit identischer Initialisierung, damit das
deterministische EM-Verfahren exakt reproduzierbar vergleichbar ist).
"""

from dataclasses import dataclass

import numpy as np

from gm_constants import LOG_LIKELIHOOD_TOL, MAX_ITERATIONS, REG_COVAR


@dataclass(frozen=True)
class Step:
    iteration: int  # 0 = E-Schritt auf den Startparametern, danach je ein M-dann-E-Zyklus
    weights: tuple  # (pi_1, ..., pi_k)
    means: tuple  # ((x, y), ...), k Einträge
    covariances: tuple  # (((a,b),(c,d)), ...), k 2x2-Matrizen
    responsibilities: tuple  # (n, k): Zugehörigkeitswahrscheinlichkeit jedes Punkts zu jeder Komponente
    log_likelihood: float
    n_changed: int  # Punkte, deren hartes Label (argmax responsibility) sich gegenüber dem Vorschritt änderte


@dataclass(frozen=True)
class RunResult:
    steps: tuple  # Step-Folge in Ausführungsreihenfolge
    converged: bool  # True, wenn die relative Log-Likelihood-Verbesserung unter die Toleranz fiel
    truncated: bool  # True, wenn max_iter erreicht wurde, ohne dass die Toleranz erreicht wurde

    @property
    def final_step(self):
        return self.steps[-1]

    @property
    def final_means(self):
        return self.final_step.means

    @property
    def final_covariances(self):
        return self.final_step.covariances

    @property
    def final_responsibilities(self):
        return self.final_step.responsibilities

    def hard_labels(self, step_index=None):
        step = self.steps[step_index] if step_index is not None else self.final_step
        resp = np.array(step.responsibilities)
        return tuple(int(l) for l in resp.argmax(axis=1))


def init_params(data, k, rng):
    """k verschiedene Datenpunkte als Startmittelwerte (wie kmeans-demos init_random),
    Startkovarianz je Komponente = globale Datenkovarianz (halbiert, damit die
    Komponenten anfangs nicht zu breit übereinanderliegen), Startgewichte gleichverteilt."""
    n = len(data)
    idx = rng.choice(n, size=k, replace=False)
    means = data[idx].copy()
    global_cov = np.cov(data, rowvar=False) if n > 1 else np.eye(2)
    if global_cov.shape != (2, 2):
        global_cov = np.eye(2)
    covariances = np.array([global_cov * 0.5 + REG_COVAR * np.eye(2) for _ in range(k)])
    weights = np.full(k, 1.0 / k)
    return weights, means, covariances


def _gaussian_logpdf(data, mean, cov):
    """Log-Dichte der multivariaten Normalverteilung, für alle Punkte vektorisiert.
    Erwartet eine bereits regularisierte Kovarianz (siehe `m_step`/`init_params`) - die
    Regularisierung wird bewusst genau EINMAL, bei der Parameterschätzung, angewendet und
    nicht hier erneut, damit sie exakt der von `sklearn.mixture.GaussianMixture`
    gespeicherten (ebenfalls regularisierten) Kovarianz entspricht - Voraussetzung für den
    exakten Kreuzvergleich in tests/test_algorithm.py."""
    d = mean.shape[0]
    _, logdet = np.linalg.slogdet(cov)
    inv_cov = np.linalg.inv(cov)
    diff = data - mean
    mahalanobis = np.einsum("ni,ij,nj->n", diff, inv_cov, diff)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + mahalanobis)


def e_step(data, weights, means, covariances):
    """Responsibilities via Bayes im Log-Raum (Log-Sum-Exp-Trick), damit keine sehr
    kleinen Dichtewerte direkt multipliziert werden. Gibt zusätzlich die
    Gesamt-Log-Likelihood der Daten unter dem aktuellen Modell zurück."""
    k = len(weights)
    n = len(data)
    log_weighted = np.empty((n, k))
    for j in range(k):
        log_weighted[:, j] = np.log(weights[j] + 1e-300) + _gaussian_logpdf(data, means[j], covariances[j])
    max_log = log_weighted.max(axis=1, keepdims=True)
    log_sum = max_log[:, 0] + np.log(np.exp(log_weighted - max_log).sum(axis=1))
    responsibilities = np.exp(log_weighted - log_sum[:, None])
    log_likelihood = float(log_sum.sum())
    return responsibilities, log_likelihood


def m_step(data, responsibilities, covariance_type):
    """Geschlossene Updates aus dem responsibility-gewichteten Mittelwert/der
    responsibility-gewichteten Streuungsmatrix - die Form der Kovarianz wird je nach
    covariance_type eingeschränkt (siehe Mathe-Abschnitt der App für alle vier Varianten)."""
    n, d = data.shape
    k = responsibilities.shape[1]
    n_k = responsibilities.sum(axis=0)
    n_k_safe = np.maximum(n_k, 1e-12)

    weights = n_k / n
    means = (responsibilities.T @ data) / n_k_safe[:, None]

    full_covariances = np.empty((k, d, d))
    for j in range(k):
        diff = data - means[j]
        weighted_diff = diff * responsibilities[:, j : j + 1]
        full_covariances[j] = (weighted_diff.T @ diff) / n_k_safe[j]

    if covariance_type == "full":
        covariances = full_covariances
    elif covariance_type == "diag":
        covariances = np.array([np.diag(np.diag(cov)) for cov in full_covariances])
    elif covariance_type == "spherical":
        covariances = np.array([np.eye(d) * (np.trace(cov) / d) for cov in full_covariances])
    elif covariance_type == "tied":
        pooled = sum(n_k[j] * full_covariances[j] for j in range(k)) / n
        covariances = np.array([pooled.copy() for _ in range(k)])
    else:
        raise ValueError(f"Unbekannter covariance_type: {covariance_type}")

    # Regularisierung genau hier (einmalig, wie sklearn.mixture.GaussianMixture), damit
    # die gespeicherten Kovarianzen nie singulär sind - siehe _gaussian_logpdf.
    covariances = covariances + REG_COVAR * np.eye(d)[None, :, :]

    return weights, means, covariances


def run(data, k, covariance_type, seed, max_iter=MAX_ITERATIONS, tol=LOG_LIKELIHOOD_TOL):
    """Führt EM vollständig protokolliert aus, ausgehend von einer frischen
    Zufalls-Initialisierung (siehe `init_params`)."""
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    weights, means, covariances = init_params(data, k, rng)
    return run_from_init(data, weights, means, covariances, covariance_type, max_iter, tol)


def run_from_init(data, weights, means, covariances, covariance_type, max_iter=MAX_ITERATIONS, tol=LOG_LIKELIHOOD_TOL):
    """Führt EM vollständig protokolliert aus, ausgehend von EXPLIZIT vorgegebenen
    Startparametern - separiert von `run`, damit Tests dieselbe Initialisierung
    identisch auch an `sklearn.mixture.GaussianMixture` (via `means_init`/
    `precisions_init`) übergeben können, für einen exakten Kreuzvergleich (EM ist bei
    gleichem Start deterministisch). Schritt 0 ist der E-Schritt auf den Startparametern,
    jeder weitere Schritt ein vollständiger M-dann-E-Zyklus. Terminiert, sobald die
    relative Verbesserung der Log-Likelihood unter `tol` fällt (EMs Log-Likelihood ist
    nachweisbar monoton nicht-fallend, siehe Mathe-Abschnitt - anders als k-Means'
    Zuweisungs-Kriterium aber i. A. KEIN exaktes Fixpunkt-Erreichen in endlich vielen
    Schritten), spätestens nach max_iter Schritten."""
    data = np.asarray(data, dtype=float)
    weights, means, covariances = np.asarray(weights), np.asarray(means), np.asarray(covariances)

    responsibilities, log_likelihood = e_step(data, weights, means, covariances)
    hard_labels = responsibilities.argmax(axis=1)
    steps = [
        Step(
            iteration=0,
            weights=tuple(float(w) for w in weights),
            means=tuple(map(tuple, means.tolist())),
            covariances=tuple(tuple(map(tuple, cov.tolist())) for cov in covariances),
            responsibilities=tuple(map(tuple, responsibilities.tolist())),
            log_likelihood=log_likelihood,
            n_changed=len(data),
        )
    ]

    converged = False
    prev_log_likelihood = log_likelihood
    for iteration in range(1, max_iter + 1):
        weights, means, covariances = m_step(data, responsibilities, covariance_type)
        responsibilities, log_likelihood = e_step(data, weights, means, covariances)
        new_hard_labels = responsibilities.argmax(axis=1)
        n_changed = int(np.sum(new_hard_labels != hard_labels))
        steps.append(
            Step(
                iteration=iteration,
                weights=tuple(float(w) for w in weights),
                means=tuple(map(tuple, means.tolist())),
                covariances=tuple(tuple(map(tuple, cov.tolist())) for cov in covariances),
                responsibilities=tuple(map(tuple, responsibilities.tolist())),
                log_likelihood=log_likelihood,
                n_changed=n_changed,
            )
        )
        hard_labels = new_hard_labels

        improvement = abs(log_likelihood - prev_log_likelihood) / max(abs(prev_log_likelihood), 1e-12)
        prev_log_likelihood = log_likelihood
        if improvement < tol:
            converged = True
            break

    truncated = not converged
    return RunResult(steps=tuple(steps), converged=converged, truncated=truncated)
