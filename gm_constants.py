"""Defaults, Regler-Grenzen, Sicherheitsgrenzen und Presets für die GMM-Demo."""

DEFAULT_N_POINTS = 150
DEFAULT_K = 3
DEFAULT_SPREAD = 0.35
DEFAULT_ELONGATION = 0.0
DEFAULT_VARIANCE_IMBALANCE = 0.0
DEFAULT_SEED = 7
DEFAULT_COVARIANCE_TYPE = "full"

N_POINTS_MIN, N_POINTS_MAX = 30, 400
K_MIN, K_MAX = 2, 6
SPREAD_MIN, SPREAD_MAX = 0.1, 0.9
ELONGATION_MIN, ELONGATION_MAX = 0.0, 1.0
VARIANCE_IMBALANCE_MIN, VARIANCE_IMBALANCE_MAX = 0.0, 1.0

# Hard safety limits so a bad slider combination can never hang the app.
MAX_ITERATIONS = 100
LOG_LIKELIHOOD_TOL = 1e-4  # relative Verbesserung, unterhalb derer EM als konvergiert gilt
REG_COVAR = 1e-6  # Diagonal-Regularisierung gegen singuläre Kovarianzmatrizen

# Fester EM-Initialisierungs-Seed für den Kovarianz-Typ-Vergleich (Kleinmultiples +
# Rand-Index-Balken) - bewusst UNABHÄNGIG vom Szenario-Seed des Reglers, damit alle vier
# Typen von einer äquivalenten Startkonfiguration aus verglichen werden und der Vergleich
# nicht zufällig von der EM-Init-Zufälligkeit des gerade eingestellten Szenario-Seeds
# verwässert wird.
COMPARISON_EM_SEED = 1

COVARIANCE_TYPES = ("full", "diag", "spherical", "tied")
COVARIANCE_TYPE_LABELS = {
    "full": "Voll (eigene Form + Rotation je Cluster)",
    "diag": "Diagonal (eigene Achsen-Streuung, keine Rotation)",
    "spherical": "Kugelförmig (eine Varianz je Cluster, isotrop)",
    "tied": "Gebunden (eine gemeinsame Form für alle Cluster)",
}

PRESETS = {
    "Einfaches Beispiel (kreisförmige Gruppen)": {
        "n_points": 90, "k": 3, "spread": 0.25, "elongation": 0.0,
        "variance_imbalance": 0.0, "seed": 2, "covariance_type": "full",
    },
    "Schwerer Fall (elliptische, rotierte Gruppen)": {
        "n_points": 200, "k": 3, "spread": 0.25, "elongation": 0.9,
        "variance_imbalance": 0.0, "seed": 3, "covariance_type": "spherical",
    },
    "Ungleich gestreute Gruppen": {
        "n_points": 200, "k": 3, "spread": 0.25, "elongation": 0.0,
        "variance_imbalance": 0.9, "seed": 1, "covariance_type": "tied",
    },
    "Viele Gruppen": {
        "n_points": 260, "k": 6, "spread": 0.3, "elongation": 0.4,
        "variance_imbalance": 0.3, "seed": 11, "covariance_type": "full",
    },
}
