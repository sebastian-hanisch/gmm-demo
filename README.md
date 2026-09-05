# Gaussian Mixture Models für weiche Sammel-Routen-Zuordnung – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-gmm-demo.streamlit.app/)**

Sechstes Stück der "Konzepte"-Reihe für die Website "Sebastian Hanisch – Operations
Research und Machine Learning", **zweiter, unabhängiger Zweig** direkt ab
[kmeans-demo](../kmeans-demo) - kein dritter Vorläufer zu HDBSCAN. Die Clustering-Linie ist
damit ein DAG mit einem Zusammenfluss UND einem zweiten, eigenständigen Ast:

```
kmeans-demo → dbscan-demo ──┐
                             ├──> hdbscan-demo
              agglomerative-demo ──────────┘
kmeans-demo → gmm-demo
```

[dbscan-demo](../dbscan-demo) und [agglomerative-demo](../agglomerative-demo) beheben
k-Means' **dichte-/verbindungsbasierte** Schwäche (Konvexitätsannahme, festes k) und
konvergieren auf [hdbscan-demo](../hdbscan-demo). Diese Demo behebt eine **andere,
unabhängige** Schwäche: k-Means nimmt an, dass alle Cluster **kugelförmig, gleich groß und
gleich gestreut** sind, und weist jeden Punkt **hart** genau einem Zentrum zu. **Gaussian
Mixture Models (GMM)** ersetzen das Zentrum durch eine volle Gauß-Verteilung (Mittelwert
**und** Kovarianzmatrix) je Cluster und die harte Zuweisung durch eine **weiche**
(wahrscheinlichkeitsbasierte) Zugehörigkeit, berechnet mit dem **EM-Algorithmus**. k-Means
ist der Grenzfall: gleiche, gegen 0 gehende, kugelförmige Varianz plus harte Zuweisung
("Hard EM").

## Warum diese Demo anders aufgebaut ist

Bei k-Means war die Schwierigkeitsachse der Zufall der Startpunkte, bei DBSCAN die Wahl von
eps/min_samples, bei agglomerativem Clustering das Linkage-Kriterium. Hier ist es die
**Kovarianz-Annahme** - voll/diagonal/kugelförmig/gebunden sind vier Ausprägungen des einen
gezeigten Verfahrens, keine vier verglichenen Methoden:

- **Einfaches Beispiel**: kreisförmige, gleich gestreute Gruppen - alle vier
  Kovarianz-Typen liefern dasselbe Ergebnis.
- **Schwerer Fall (elliptische, rotierte Gruppen)**: stark elliptische Gruppen, deren lange
  Achse gezielt auf den jeweils nächsten Cluster zeigt - **kugelförmige** (k-Means-artige)
  Kovarianz scheitert (Rand-Index ≈0.56), **volle** Kovarianz löst es sauber (≈0.97). Live
  in der "📐"-Sektion nachgewiesen, nicht nur behauptet.
- **Ungleich gestreute Gruppen**: eine Gruppe deutlich diffuser als die übrigen (weiterhin
  kreisförmig) - eine ZWEITE, unabhängige k-Means-Schwäche: **gebundene** Kovarianz
  (eine gemeinsame Form für alle Cluster) erzwingt implizit gleiche Varianz überall und
  scheitert (≈0.72), während Typen mit eigener Varianz je Komponente
  (spherical/diag/full) das Ungleichgewicht problemlos abbilden (≈0.98).
- **Viele Gruppen**: mehr Komponenten gleichzeitig, wachsende Modellkomplexität.

## Visualisierung

Zwei in dieser Reihe neue Visualisierungen: die Punktfarbe ist eine **Mischung** der
Komponentenfarben nach Responsibility (weiche statt harter Zuordnung), und **Kovarianz-
Ellipsen** zeigen die aktuell geschätzte Form jeder Komponente. Der Schritt-Regler
durchläuft EM-Iterationen (wie kmeans-demos Lloyd-Iterationen), begleitet von einem
Log-Likelihood-Diagramm, das die EM-Monotonie-Eigenschaft live zeigt. Die
"Und mit anderen Kovarianz-Typen?"-Kleinmultiples vergleichen alle vier Typen bei
Konvergenz - mit einer FESTEN EM-Startkonfiguration (unabhängig vom Szenario-Seed), damit
alle vier Typen von einer äquivalenten Ausgangslage aus verglichen werden.

## Sicherheitsgrenzen

`MAX_ITERATIONS` (100) verhindert eine Endlosschleife. Anders als bei k-Means' Zuweisungs-
Kriterium gibt es bei EM i. A. **kein** exaktes Fixpunkt-Erreichen in endlich vielen
Schritten - terminiert wird, sobald sich die Log-Likelihood nur noch um weniger als
`LOG_LIKELIHOOD_TOL` verbessert. Eine kleine Diagonal-Regularisierung (`REG_COVAR`, wie
sklearns Default) verhindert singuläre Kovarianzmatrizen.

## Verifikation

- **Handgerechnete Beispiele**: E-Schritt (Responsibility zweier Gauß-Komponenten für
  einen Punkt) und M-Schritt (gewichteter Mittelwert/Kovarianz aus zwei Punkten) von Hand
  nachgerechnet.
- **Exakter Kreuzvergleich mit scikit-learn**: `sklearn.mixture.GaussianMixture`, gestartet
  mit NUMERISCH IDENTISCHEN Anfangsparametern (`means_init`/`precisions_init`) für alle
  vier Kovarianz-Typen - EM ist bei gleichem Start deterministisch, daher werden finale
  Mittelwerte/Kovarianzen/Log-Likelihood exakt (nicht nur toleranzbasiert) verglichen
  (scikit-learn ist ausschließlich ein Test-Dependency, kein Laufzeit-Dependency der App).
- **Monotonie-Eigenschaft**: Log-Likelihood fällt über viele Zufallsinstanzen und alle vier
  Kovarianz-Typen hinweg nie.
- **Struktur-Invarianten**: Responsibilities je Punkt summieren zu 1, Gewichte summieren zu
  1, alle Kovarianzen sind positiv semidefinit.
- **Kern-Behauptungen der Demo direkt getestet**: volle Kovarianz schlägt kugelförmige
  Kovarianz bei elliptischen, rotierten Gruppen deutlich
  (`test_full_covariance_beats_spherical_on_elongated_rotated_clusters`); gebundene
  Kovarianz scheitert bei ungleicher Streuung, wo die übrigen Typen erfolgreich sind
  (`test_tied_covariance_fails_on_unequal_variance_where_others_succeed`).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Hauptablauf: Presets, Einstellungen, EM-Animation, Kleinmultiples, Rand-Index-Vergleich, Formulierungs-Expander |
| `gm_constants.py` | Defaults, Regler-Grenzen, Sicherheitsgrenzen, `PRESETS`, `COVARIANCE_TYPES` |
| `gm_presets.py` | `SettingSpec`/`SETTING_SPECS`, Permalink-Logik, Presets, Zufalls-Seed-Button |
| `gm_scenario.py` | Zufällige Punktwolken: Gauß-Gruppen auf einem Ring, mit unabhängig einstellbarer Elongation (elliptische Form + gezielte Rotation) und Varianz-Ungleichgewicht |
| `gm_algorithm.py` | EM-Algorithmus from scratch (E-/M-Schritt, alle vier Kovarianz-Typen), mit vollständigem Iterations-Protokoll |
| `gm_evaluation.py` | Rand-Index (from scratch), Kovarianz-Typ-Vergleich bei Konvergenz |
| `gm_visualization.py` | Punktwolke mit weicher Farbmischung + Kovarianz-Ellipsen, Kleinmultiples, Log-Likelihood- und Rand-Index-Diagramm (Plotly) |
| `tests/` | Handinstanzen, exakter sklearn-Kreuzvergleich, Monotonie, Struktur-Invarianten, Szenario-Reproduzierbarkeit, die beiden zentralen Kovarianz-Typ-Nachweise |

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

---

Teil des [Operations-Research-Demo-Portfolios](https://sebastianhanisch.net/demos.html) von
[Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning.
Interesse an einer maßgeschneiderten Lösung? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html).
