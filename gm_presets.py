"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster aus
dem OR-Demo-Portfolio, siehe ag_presets.py in agglomerative-demo)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import gm_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


def _covariance_type_caster(v):
    return v if v in C.COVARIANCE_TYPES else C.DEFAULT_COVARIANCE_TYPE


SETTING_SPECS = {
    "n_points_slider": SettingSpec("n", int, C.DEFAULT_N_POINTS, C.N_POINTS_MIN, C.N_POINTS_MAX),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX),
    "spread_slider": SettingSpec("spread", float, C.DEFAULT_SPREAD, C.SPREAD_MIN, C.SPREAD_MAX),
    "elongation_slider": SettingSpec(
        "elong", float, C.DEFAULT_ELONGATION, C.ELONGATION_MIN, C.ELONGATION_MAX
    ),
    "variance_imbalance_slider": SettingSpec(
        "vimb", float, C.DEFAULT_VARIANCE_IMBALANCE, C.VARIANCE_IMBALANCE_MIN, C.VARIANCE_IMBALANCE_MAX
    ),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, 2_000_000_000),
    "covariance_type_radio": SettingSpec("cov", _covariance_type_caster, C.DEFAULT_COVARIANCE_TYPE),
}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                if spec.lo is not None:
                    value = max(spec.lo, value)
                if spec.hi is not None:
                    value = min(spec.hi, value)
                st.session_state[state_key] = value
            except (ValueError, TypeError):
                pass
    st.session_state["permalink_loaded"] = True


def sync_query_params(n_points, k, spread, elongation, variance_imbalance, seed, covariance_type):
    try:
        st.query_params["n"] = str(int(n_points))
        st.query_params["k"] = str(int(k))
        st.query_params["spread"] = str(spread)
        st.query_params["elong"] = str(elongation)
        st.query_params["vimb"] = str(variance_imbalance)
        st.query_params["seed"] = str(int(seed))
        st.query_params["cov"] = covariance_type
    except Exception:
        pass


def apply_preset(name):
    p = C.PRESETS[name]
    st.session_state["n_points_slider"] = p["n_points"]
    st.session_state["k_slider"] = p["k"]
    st.session_state["spread_slider"] = p["spread"]
    st.session_state["elongation_slider"] = p["elongation"]
    st.session_state["variance_imbalance_slider"] = p["variance_imbalance"]
    st.session_state["covariance_type_radio"] = p["covariance_type"]
    st.session_state["seed_input"] = p["seed"]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, 2_000_000_000)
