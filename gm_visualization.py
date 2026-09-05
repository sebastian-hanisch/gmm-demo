"""Plotly-Visualisierungen: Punktwolke mit WEICHER Farbmischung nach Responsibility plus
Kovarianz-Ellipsen je Komponente (beides neu gegenüber den harten Cluster-Farben der
übrigen Demos), Kleinmultiples je Kovarianz-Typ, Log-Likelihood-Verlauf und
Rand-Index-Balkendiagramm."""

import numpy as np

from gm_constants import COVARIANCE_TYPE_LABELS, COVARIANCE_TYPES

CLUSTER_PALETTE = [
    "#1f77b4", "#d68a2e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def _hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([int(hex_color[i : i + 2], 16) for i in (0, 2, 4)], dtype=float)


def _rgb_to_hex(rgb):
    r, g, b = (int(max(0, min(255, round(c)))) for c in rgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def _blended_colors(responsibilities, k):
    palette_rgb = np.array([_hex_to_rgb(CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)]) for i in range(k)])
    blended_rgb = responsibilities @ palette_rgb
    return [_rgb_to_hex(c) for c in blended_rgb]


def _covariance_ellipse_points(mean, cov, n_std=2.0, n_points=60):
    """Kontur bei n_std Standardabweichungen entlang der Hauptachsen der Kovarianz -
    Eigenzerlegung liefert Richtung (Eigenvektoren) und Länge (Wurzel der Eigenwerte)
    der Ellipsenachsen."""
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.stack([np.cos(theta), np.sin(theta)])
    scaled = eigenvectors @ (n_std * np.sqrt(eigenvalues)[:, None] * circle)
    return mean[0] + scaled[0], mean[1] + scaled[1]


def _axis_range(data):
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    padx = (xmax - xmin) * 0.15 or 1.0
    pady = (ymax - ymin) * 0.15 or 1.0
    return [xmin - padx, xmax + padx], [ymin - pady, ymax + pady]


def _build_figure(data, step, height, legend, ellipse_width):
    import plotly.graph_objects as go

    responsibilities = np.array(step.responsibilities)
    means = np.array(step.means)
    covariances = np.array(step.covariances)
    k = means.shape[0]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data[:, 0], y=data[:, 1], mode="markers", showlegend=False, hoverinfo="skip",
            marker=dict(
                color=_blended_colors(responsibilities, k), size=7, line=dict(width=0.5, color="white")
            ),
        )
    )
    for j in range(k):
        color = CLUSTER_PALETTE[j % len(CLUSTER_PALETTE)]
        ex, ey = _covariance_ellipse_points(means[j], covariances[j])
        fig.add_trace(
            go.Scatter(
                x=ex, y=ey, mode="lines", line=dict(color=color, width=ellipse_width),
                hoverinfo="skip", showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[means[j][0]], y=[means[j][1]], mode="markers",
                marker=dict(color=color, size=11, symbol="x", line=dict(width=2, color="white")),
                name=f"Komponente {j + 1}", showlegend=legend, hoverinfo="skip",
            )
        )

    xr, yr = _axis_range(data)
    fig.update_layout(
        template="plotly_white", height=height,
        xaxis=dict(visible=False, range=xr, fixedrange=True),
        yaxis=dict(visible=False, range=yr, fixedrange=True, scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(t=40 if legend else 5, l=10, r=10, b=10),
    )
    return fig


def build_scatter_figure(data, result, step_index):
    return _build_figure(data, result.steps[step_index], height=460, legend=True, ellipse_width=2.5)


def build_mini_scatter_figure(data, result, step_index=None):
    step = result.final_step if step_index is None else result.steps[step_index]
    return _build_figure(data, step, height=220, legend=False, ellipse_width=2.0)


def build_loglik_chart(result):
    import plotly.graph_objects as go

    iterations = [s.iteration for s in result.steps]
    log_likelihoods = [s.log_likelihood for s in result.steps]
    fig = go.Figure(
        go.Scatter(x=iterations, y=log_likelihoods, mode="lines+markers", line=dict(color="#1f77b4"))
    )
    fig.update_layout(
        template="plotly_white", height=220,
        xaxis=dict(title="Iteration", fixedrange=True),
        yaxis=dict(title="Log-Likelihood", fixedrange=True, tickformat=",.0f"),
        margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    return fig


def build_rand_index_bar_chart(scores):
    import plotly.graph_objects as go

    fig = go.Figure(
        go.Bar(
            x=[COVARIANCE_TYPE_LABELS[c] for c in COVARIANCE_TYPES],
            y=[scores[c] for c in COVARIANCE_TYPES],
            marker_color=[CLUSTER_PALETTE[i % len(CLUSTER_PALETTE)] for i in range(len(COVARIANCE_TYPES))],
        )
    )
    fig.update_layout(
        template="plotly_white", height=280,
        yaxis=dict(title="Rand-Index", range=[0, 1.05], fixedrange=True),
        xaxis=dict(fixedrange=True), margin=dict(t=20, l=10, r=10, b=10), showlegend=False,
    )
    return fig
