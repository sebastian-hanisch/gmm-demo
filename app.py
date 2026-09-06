"""Gaussian Mixture Models für weiche Sammel-Routen-Zuordnung - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Anders als die Fall-Demos im Portfolio (ein Anwendungsfall, mehrere Verfahren im
Vergleich) zeigt diese Demo EIN Verfahren - Gaussian Mixture Models (EM-Algorithmus) -
und lässt stattdessen die Empfindlichkeit gegenüber der **Kovarianz-Annahme** wachsen: von
kreisförmigen Gruppen, bei denen jede Annahme zum selben Ergebnis führt, bis zu
elliptischen, rotierten oder unterschiedlich gestreuten Gruppen, an denen k-Means-artige
Annahmen (kugelförmig, gleiche Varianz, harte Zuweisung) nachweislich scheitern. Sechstes
Stück der "Konzepte"-Reihe, zweiter (unabhängiger) Zweig ab kmeans-demo neben der
DBSCAN/Agglomerativ/HDBSCAN-Linie (siehe README für die Einordnung).

Lauffähig mit: streamlit run app.py
"""

import time

import streamlit as st

import gm_constants as C
from gm_algorithm import run
from gm_evaluation import covariance_type_comparison
from gm_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from gm_scenario import generate_instance
from gm_visualization import (
    build_loglik_chart,
    build_mini_scatter_figure,
    build_rand_index_bar_chart,
    build_scatter_figure,
)

st.set_page_config(page_title="Gaussian Mixture Models – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _compute_run(n_points, k, spread, elongation, variance_imbalance, seed, covariance_type):
    instance = generate_instance(n_points, k, spread, elongation, variance_imbalance, seed)
    result = run(instance.as_array(), k, covariance_type, seed)
    return instance, result


@st.cache_data(show_spinner=False)
def _compute_mini_run(instance, covariance_type):
    return run(instance.as_array(), instance.k, covariance_type, C.COMPARISON_EM_SEED)


@st.cache_data(show_spinner=False)
def _compute_covariance_comparison(instance):
    return covariance_type_comparison(
        instance.as_array(), instance.true_labels, instance.k, C.COMPARISON_EM_SEED
    )


st.title("🌐 Gaussian Mixture Models für weiche Sammel-Routen-Zuordnung")
st.markdown(
    """
Lieferadressen sollen zu Sammel-Routen gruppiert werden - aber nicht jede Adresse gehört
eindeutig zu genau einer Route: manche liegen genau zwischen zwei Routengebieten. **k-Means**
weist trotzdem jede Adresse **hart** genau einem Zentrum zu und nimmt an, dass alle
Routengebiete gleich groß und kreisförmig sind. **Gaussian Mixture Models (GMM)** ersetzen
das Zentrum durch eine volle Gauß-Verteilung (Mittelwert **und** Kovarianzmatrix) je Gebiet
und die harte Zuweisung durch eine **weiche**, wahrscheinlichkeitsbasierte Zugehörigkeit -
berechnet mit dem **EM-Algorithmus**. Genau **wie** das funktioniert, erklärt der
aufgeklappte Abschnitt direkt darunter - bevor weiter unten die Iterationen live ablaufen
und die Frage "📐 Was bringt volle Kovarianz gegenüber k-Means-artigen Annahmen?" live
beantwortet wird.
"""
)
st.caption(
    "Anders als die Fall-Demos im Portfolio, die an einem Anwendungsfall mehrere Verfahren "
    "vergleichen, zeigt diese Demo - Teil der wachsenden \"Konzepte\"-Reihe - **ein** "
    "Verfahren an einem wachsenden Beispiel: nicht Zufall der Startpunkte (wie bei k-Means) "
    "oder ein Suchradius (wie bei DBSCAN) ist hier die Schwierigkeitsachse, sondern die "
    "gewählte **Kovarianz-Annahme** - ein Parameter des einen gezeigten Verfahrens, keine "
    "vier verglichenen Methoden."
)

with st.expander("So funktioniert der EM-Algorithmus", expanded=True):
    st.markdown(
        """
Der EM-Algorithmus (Expectation-Maximization) wiederholt zwei Schritte, bis sich die
Log-Likelihood der Daten nicht mehr nennenswert verbessert:

1. **E-Schritt (Expectation)**: für jeden Punkt wird die **Responsibility** - die
   Wahrscheinlichkeit, zu jeder Komponente zu gehören - über den Satz von Bayes aus den
   aktuellen Gauß-Verteilungen berechnet. Anders als bei k-Means ist das eine **weiche**
   Zuordnung: ein Punkt kann z. B. zu 70% Komponente 1 und zu 30% Komponente 2 gehören.
2. **M-Schritt (Maximization)**: jede Komponente bekommt einen neuen Mittelwert und eine
   neue Kovarianzmatrix - beide als Responsibility-gewichteter Durchschnitt über alle
   Punkte (nicht nur die "eigenen", wie bei k-Means).

Die **Kovarianz-Annahme** legt fest, wie viel Form jede Komponente annehmen darf:

- **Voll**: eigene, frei rotierbare/elliptische Form je Komponente - flexibelste Variante.
- **Diagonal**: eigene Form je Komponente, aber immer achsenparallel (keine Rotation).
- **Kugelförmig**: eine einzige Varianz je Komponente (isotrop) - das ist die Annahme, die
  k-Means implizit trifft, nur mit weicher statt harter Zuweisung.
- **Gebunden**: alle Komponenten teilen sich EINE gemeinsame Form.

Die Punktwolke weiter unten zeigt das live: die Punktfarbe ist eine **Mischung** der
Komponentenfarben nach Responsibility (nicht eine einzelne harte Farbe), die Ellipsen
zeigen die aktuell geschätzte Form jeder Komponente.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
PRESET_HELP = {
    "Einfaches Beispiel (kreisförmige Gruppen)": "Keine Elongation, keine Varianz-Unterschiede - alle Kovarianz-Typen liefern dasselbe Ergebnis.",
    "Schwerer Fall (elliptische, rotierte Gruppen)": "Stark elliptische Gruppen mit unterschiedlicher Rotation - kugelförmige (k-Means-artige) Kovarianz scheitert, volle Kovarianz löst es sauber.",
    "Ungleich gestreute Gruppen": "Eine Gruppe deutlich diffuser als die übrigen - gebundene Kovarianz (eine gemeinsame Form für alle) scheitert hier, weil sie implizit gleiche Varianz erzwingt.",
    "Viele Gruppen": "Mehr Komponenten gleichzeitig - zeigt wachsende Modellkomplexität.",
}
preset_cols = st.columns(len(C.PRESETS))
for i, name in enumerate(C.PRESETS.keys()):
    with preset_cols[i]:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP[name])

st.caption(
    "🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, "
    "um ein Szenario zu teilen."
)

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_points = st.slider("Anzahl Adressen", *bounds("n_points_slider"), key="n_points_slider")
    k = st.slider("Anzahl Routengebiete (k)", *bounds("k_slider"), key="k_slider")
    spread = st.slider(
        "Basis-Streuung", *bounds("spread_slider"), key="spread_slider", step=0.05,
        help="Klein = Gruppen klar getrennt. Groß = Gruppen überlappen sich spürbar.",
    )
    elongation = st.slider(
        "Elongation (Ellipsen-Form)", *bounds("elongation_slider"), key="elongation_slider", step=0.05,
        help="0 = alle Gruppen kreisförmig. Hoch = Gruppen werden elliptisch, jede mit einer "
        "eigenen, zufälligen Rotation - die Achse, an der kugelförmige Kovarianz scheitert.",
    )
    variance_imbalance = st.slider(
        "Varianz-Ungleichgewicht", *bounds("variance_imbalance_slider"), key="variance_imbalance_slider",
        step=0.05,
        help="0 = alle Gruppen gleich stark gestreut. Hoch = eine Gruppe wird deutlich "
        "diffuser als die übrigen - die Achse, an der gebundene Kovarianz scheitert.",
    )
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1)

    st.markdown("**GMM-Parameter**")
    covariance_type = st.radio(
        "Kovarianz-Annahme (für die Animation unten)",
        options=C.COVARIANCE_TYPES, key="covariance_type_radio",
        format_func=lambda c: C.COVARIANCE_TYPE_LABELS[c],
        help="Steuert die Primäransicht unten - der '📐'-Vergleich weiter unten prüft "
        "ohnehin immer alle vier Kovarianz-Typen parallel.",
    )

    st.button(
        "🎲 Neue Punktwolke generieren",
        width="stretch",
        on_click=randomize_seed,
        help="Würfelt einen neuen Zufalls-Seed für die Adressstandorte.",
    )

sync_query_params(n_points, k, spread, elongation, variance_imbalance, seed, covariance_type)

with st.spinner("Führe EM-Algorithmus aus..."):
    instance, result = _compute_run(
        int(n_points), int(k), spread, elongation, variance_imbalance, int(seed), covariance_type
    )

max_step = len(result.steps) - 1
run_key = (n_points, k, spread, elongation, variance_imbalance, seed, covariance_type)
if "gm_step" not in st.session_state or st.session_state.get("gm_step_owner") != run_key:
    st.session_state["gm_step"] = max_step
    st.session_state["gm_step_owner"] = run_key

st.markdown("## 🎯 EM-Algorithmus in Aktion")

step_col, play_col = st.columns([5, 1])
with step_col:
    if max_step == 0:
        step = 0
        st.caption("Bereits im Startzustand konvergiert - kein Regler nötig.")
    else:
        step = st.slider(
            "Schritt (Iteration)", 0, max_step, key="gm_step",
            help="Schritt 0 = E-Schritt auf den Startparametern, danach je ein vollständiger "
            "M-dann-E-Zyklus.",
        )
with play_col:
    auto_play = st.button("▶️ Abspielen", width="stretch")

scatter_slot = st.empty()
loglik_slot = st.empty()


def _render(current_step):
    scatter_slot.plotly_chart(
        build_scatter_figure(instance.as_array(), result, current_step),
        width="stretch", key=f"scatter_{current_step}",
    )
    loglik_slot.plotly_chart(
        build_loglik_chart(result), width="stretch", key=f"loglik_{current_step}",
    )


if auto_play:
    n_frames = min(max_step + 1, 30)
    frame_skip = max(1, (max_step + 1) // n_frames)
    for s in list(range(0, max_step, frame_skip)) + [max_step]:
        _render(s)
        time.sleep(0.35)
    step = max_step
else:
    _render(step)

current = result.steps[step]
lm1, lm2, lm3, lm4 = st.columns(4)
lm1.metric("Iteration", current.iteration)
lm2.metric(
    "Log-Likelihood", f"{current.log_likelihood:,.1f}",
    help="Log-Wahrscheinlichkeit der Daten unter dem aktuellen Modell - die Größe, die EM "
    "in jedem Schritt erhöht (oder gleich lässt), nie verringert.",
)
lm3.metric(
    "Adressen mit geändertem harten Label", current.n_changed,
    help="Hartes Label = Komponente mit der höchsten Responsibility. 0 bedeutet: stabil.",
)
lm4.metric("Konvergiert?", "Ja" if result.converged else "Nein")

if result.truncated:
    st.error(
        f"⛔ Nach {C.MAX_ITERATIONS} Iterationen noch nicht konvergiert - das gezeigte "
        f"Ergebnis ist der Zwischenstand, nicht der stabile Endzustand."
    )

st.markdown("**Und mit anderen Kovarianz-Typen?**")
st.caption(
    "Gleiche Adressen wie oben, jeweils mit derselben EM-Startkonfiguration - nur die "
    "Kovarianz-Annahme unterscheidet sich, jeweils am Konvergenzpunkt gezeigt."
)
example_cols = st.columns(len(C.COVARIANCE_TYPES))
for col, example_type in zip(example_cols, C.COVARIANCE_TYPES):
    with col:
        example_result = _compute_mini_run(instance, example_type)
        st.plotly_chart(
            build_mini_scatter_figure(instance.as_array(), example_result),
            width="stretch", key=f"mini_{example_type}",
        )
        st.caption(C.COVARIANCE_TYPE_LABELS[example_type])

st.markdown("---")

st.subheader("📐 Was bringt volle Kovarianz gegenüber k-Means-artigen Annahmen?")
st.markdown(
    """
Live für Ihr aktuelles Szenario berechnet, nicht nur behauptet: der **Rand-Index** (Anteil
der Punktpaare, bei denen die berechnete Aufteilung mit der tatsächlichen
Gruppenzugehörigkeit übereinstimmt - 1.0 = perfekte Übereinstimmung, ~0.5 = kaum besser als
Zufall) für alle vier Kovarianz-Typen, jeweils bei Konvergenz:
"""
)

scores = _compute_covariance_comparison(instance)
st.plotly_chart(build_rand_index_bar_chart(scores), width="stretch", key="rand_index_bar")

gap = max(scores.values()) - min(scores.values())
if gap > 0.2:
    worst = min(scores, key=scores.get)
    best = max(scores, key=scores.get)
    st.success(
        f"✅ Bei diesem Szenario erreicht {C.COVARIANCE_TYPE_LABELS[worst]} nur einen "
        f"Rand-Index von {scores[worst]:.2f}, während {C.COVARIANCE_TYPE_LABELS[best]} bei "
        f"{scores[best]:.2f} liegt - dasselbe Datenset, nur die Kovarianz-Annahme "
        f"unterscheidet sich."
    )
else:
    st.info(
        "Bei diesem Szenario liegen alle vier Kovarianz-Typen noch nah beieinander - mehr "
        "Elongation oder Varianz-Ungleichgewicht (Regler links) macht den Unterschied "
        "deutlicher."
    )

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Mischungsdichte**: ein GMM modelliert die Daten als gewichtete Summe von $k$
Gauß-Verteilungen:

$$
p(x) = \sum_{j=1}^{k} \pi_j \, \mathcal{N}(x \mid \mu_j, \Sigma_j), \qquad \sum_j \pi_j = 1
$$

Gesucht sind die Parameter $\pi_j, \mu_j, \Sigma_j$, die die Log-Likelihood der Daten
maximieren - dafür gibt es (anders als bei k-Means' Zuweisungsschritt) keine geschlossene
Lösung in einem Schritt, weshalb der EM-Algorithmus iterativ vorgeht.

**E-Schritt**: Responsibility von Punkt $x_i$ für Komponente $j$, über Bayes:

$$
r_{ij} = \frac{\pi_j \, \mathcal{N}(x_i \mid \mu_j, \Sigma_j)}
{\sum_{l=1}^{k} \pi_l \, \mathcal{N}(x_i \mid \mu_l, \Sigma_l)}
$$

**M-Schritt**: geschlossene, Responsibility-gewichtete Updates (mit $n_j = \sum_i r_{ij}$):

$$
\pi_j \leftarrow \frac{n_j}{n}, \qquad
\mu_j \leftarrow \frac{1}{n_j} \sum_i r_{ij} \, x_i, \qquad
\Sigma_j \leftarrow \frac{1}{n_j} \sum_i r_{ij} \, (x_i - \mu_j)(x_i - \mu_j)^\top
$$

Die **Kovarianz-Annahme** schränkt die Form von $\Sigma_j$ ein: voll (frei), diagonal
(achsenparallel), kugelförmig ($\Sigma_j = \sigma_j^2 I$, eine Zahl statt einer Matrix) oder
gebunden ($\Sigma_j = \Sigma$ für alle $j$).

**Monotonie**: jeder EM-Schritt kann die Log-Likelihood der Daten nur vergrößern oder
gleich lassen, nie verkleinern (Dempster, Laird & Rubin, 1977) - siehe
`tests/test_algorithm.py`, das das über viele Zufallsinstanzen prüft. Anders als bei
k-Means' Zuweisungskriterium gibt es hier aber i. A. **kein** exaktes Fixpunkt-Erreichen in
endlich vielen Schritten - deshalb terminiert diese Demo, sobald sich die Log-Likelihood
nur noch um weniger als eine feste Toleranz verbessert, nicht erst bei exakter Gleichheit.

**k-Means als Grenzfall**: schrumpft man alle Kovarianzen gegen dieselbe, gegen Null gehende
kugelförmige Matrix und ersetzt die weichen Responsibilities durch eine harte 0/1-Zuordnung
zur wahrscheinlichsten Komponente ("Hard EM"), reduziert sich der M-Schritt exakt auf
Lloyd's Algorithmus - k-Means ist ein GMM mit den am stärksten eingeschränkten Annahmen.

**Möglicher nächster Schritt** (hier nicht gebaut): die Anzahl der Komponenten $k$ selbst
automatisch wählen, z. B. über das Bayessche Informationskriterium (BIC), das
Modellkomplexität gegen Anpassungsgüte abwägt.

Implementiert in `gm_algorithm.py` (EM-Algorithmus, alle vier Kovarianz-Typen) und
`gm_evaluation.py` (Rand-Index, Kovarianz-Typ-Vergleich).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
