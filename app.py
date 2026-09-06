import datetime
from scipy.optimize import minimize
from scipy.stats import genpareto, norm
import numpy as np
import pandas as pd
# Imports for exact US Trading Day calculations
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    nearest_workday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ==========================================
# 1. CORE MATH & LEGS LOGIC
# ==========================================


class USTradingCalendar(AbstractHolidayCalendar):
    """Custom holiday calendar defining exact US market holidays (NYSE/NASDAQ)."""

    rules = [
        Holiday("NewYearsDay", month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, observance=nearest_workday),
        Holiday("IndependenceDay", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas", month=12, day=25, observance=nearest_workday),
    ]


def calculate_payoff(S_T: np.ndarray, legs: list) -> np.ndarray:
    total_payoff = np.zeros_like(S_T)
    for leg in legs:
        if leg["type"] == "call":
            payoff = np.maximum(S_T - leg["strike"], 0)
        elif leg["type"] == "put":
            payoff = np.maximum(leg["strike"] - S_T, 0)
        total_payoff += leg["pos"] * payoff
    return total_payoff


def calculate_structure_greek(
    S: np.ndarray,
    legs: list,
    T: float,
    r: float,
    q: float,
    sigma: float,
    greek: str,
) -> np.ndarray:
    """Vectorized calculation of total structure Value & Greeks across an array of spot prices.

    Uses standard Black-Scholes partial derivatives and Risk Neutral Pricing.
    Includes secondary cross-derivatives for advanced risk profiling.
    """
    total_val = np.zeros_like(S, dtype=float)
    T_safe = max(T, 1e-5)  # Prevent division by zero near expiration

    for leg in legs:
        K = leg["strike"]
        pos = leg["pos"]
        type_ = leg["type"]

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T_safe) / (
            sigma * np.sqrt(T_safe)
        )
        d2 = d1 - sigma * np.sqrt(T_safe)

        n_d1 = norm.pdf(d1)
        N_d1 = norm.cdf(d1)
        N_d2 = norm.cdf(d2)
        N_neg_d1 = norm.cdf(-d1)
        N_neg_d2 = norm.cdf(-d2)

        val = np.zeros_like(S)

        # Calculate Option Premium (Risk Neutral Value) or its respective Greeks
        if greek == "BS Value":
            if type_ == "call":
                val = (
                    S * np.exp(-q * T_safe) * N_d1
                    - K * np.exp(-r * T_safe) * N_d2
                )
            else:
                val = (
                    K * np.exp(-r * T_safe) * N_neg_d2
                    - S * np.exp(-q * T_safe) * N_neg_d1
                )
        elif greek == "Delta":
            if type_ == "call":
                val = np.exp(-q * T_safe) * N_d1
            else:
                val = np.exp(-q * T_safe) * (N_d1 - 1)
        elif greek == "Gamma":
            val = (np.exp(-q * T_safe) * n_d1) / (S * sigma * np.sqrt(T_safe))
        elif greek == "Vega":
            val = (
                S * np.exp(-q * T_safe) * n_d1 * np.sqrt(T_safe)
            ) / 100.0  # Per 1% change
        elif greek == "Theta":
            term1 = -(S * n_d1 * sigma * np.exp(-q * T_safe)) / (
                2 * np.sqrt(T_safe)
            )
            if type_ == "call":
                val = (
                    term1
                    - r * K * np.exp(-r * T_safe) * N_d2
                    + q * S * np.exp(-q * T_safe) * N_d1
                ) / 365.0
            else:
                val = (
                    term1
                    + r * K * np.exp(-r * T_safe) * N_neg_d2
                    - q * S * np.exp(-q * T_safe) * N_neg_d1
                ) / 365.0
        elif greek == "Rho":
            if type_ == "call":
                val = (K * T_safe * np.exp(-r * T_safe) * N_d2) / 100.0
            else:
                val = (-K * T_safe * np.exp(-r * T_safe) * N_neg_d2) / 100.0
        elif greek == "Vomma":
            # dVega/dVol = Vega * d1 * d2 / sigma
            vega_raw = S * np.exp(-q * T_safe) * n_d1 * np.sqrt(T_safe)
            val = (
                vega_raw * d1 * d2 / sigma
            ) / 10000.0  # Scaled for 1% x 1% convexity mapping
        elif greek == "Vanna":
            # dDelta/dVol = -exp(-qT) * n(d1) * d2 / sigma
            val = (
                -np.exp(-q * T_safe) * n_d1 * d2 / sigma
            ) / 100.0  # Per 1% Vol change mapping to Delta

        total_val += pos * val

    return total_val


def value_option_black_scholes(
    S_0: float,
    legs: list,
    T: float,
    r: float,
    mu: float,
    sigma: float,
    q: float = 0.0,
    num_simulations: int = 10000,
) -> tuple:
    total_val = 0.0
    for leg in legs:
        K = leg["strike"]
        d1 = (np.log(S_0 / K) + (mu - q + 0.5 * sigma**2) * T) / (
            sigma * np.sqrt(T)
        )
        d2 = d1 - sigma * np.sqrt(T)

        if leg["type"] == "call":
            val = (S_0 * np.exp((mu - q) * T) * norm.cdf(d1)) - (
                K * norm.cdf(d2)
            )
        else:
            val = (K * norm.cdf(-d2)) - (
                S_0 * np.exp((mu - q) * T) * norm.cdf(-d1)
            )
        total_val += leg["pos"] * val

    expected_value = float(np.exp(-r * T) * total_val)

    Z = np.random.standard_normal(num_simulations)
    S_T = S_0 * np.exp((mu - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs = calculate_payoff(S_T, legs)

    return expected_value, S_T, payoffs


def value_option_bootstrap_volatility(
    S_0: float,
    legs: list,
    T: float,
    r: float,
    mu: float,
    hist_vol_array: np.ndarray,
    q: float = 0.0,
    num_simulations: int = 10000,
) -> tuple:
    sampled_vols = np.random.choice(
        hist_vol_array, size=num_simulations, replace=True
    )
    simulated_values = np.zeros(num_simulations)

    for leg in legs:
        K = leg["strike"]
        d1 = (np.log(S_0 / K) + (mu - q + 0.5 * sampled_vols**2) * T) / (
            sampled_vols * np.sqrt(T)
        )
        d2 = d1 - sampled_vols * np.sqrt(T)

        if leg["type"] == "call":
            vals = (S_0 * np.exp((mu - q) * T) * norm.cdf(d1)) - (
                K * norm.cdf(d2)
            )
        else:
            vals = (K * norm.cdf(-d2)) - (
                S_0 * np.exp((mu - q) * T) * norm.cdf(-d1)
            )

        simulated_values += leg["pos"] * vals

    expected_value = float(np.mean(simulated_values) * np.exp(-r * T))

    Z = np.random.standard_normal(num_simulations)
    S_T = S_0 * np.exp(
        (mu - q - 0.5 * sampled_vols**2) * T + sampled_vols * np.sqrt(T) * Z
    )
    payoffs = calculate_payoff(S_T, legs)

    return expected_value, S_T, payoffs, sampled_vols


def fit_gpd(
    returns: np.ndarray, percentile: float = 5, tail: str = "left"
) -> tuple:
    if tail == "left":
        threshold = float(np.percentile(returns, percentile))
        exceedances = threshold - returns[returns < threshold]
    elif tail == "right":
        threshold = float(np.percentile(returns, 100 - percentile))
        exceedances = returns[returns > threshold] - threshold

    if len(exceedances) == 0:
        return threshold, 0.001, 0.001

    shape, loc, scale = genpareto.fit(exceedances, floc=0)
    return float(threshold), float(shape), float(scale)


def value_option_evt_two_tailed(
    S_0: float,
    legs: list,
    T: float,
    r: float,
    mu: float,
    hist_returns: np.ndarray,
    q: float = 0.0,
    tail_percentile: float = 5,
    num_simulations: int = 10000,
    trading_days: int = 30,
) -> tuple:
    dt = 1 / 252

    l_thresh, l_xi, l_beta = fit_gpd(
        hist_returns, percentile=tail_percentile, tail="left"
    )
    r_thresh, r_xi, r_beta = fit_gpd(
        hist_returns, percentile=tail_percentile, tail="right"
    )

    center_returns = hist_returns[
        (hist_returns >= l_thresh) & (hist_returns <= r_thresh)
    ]
    if len(center_returns) == 0:
        center_returns = hist_returns

    historical_mean = float(np.mean(hist_returns))
    center_returns = center_returns - historical_mean

    S_T = np.zeros(num_simulations)

    for i in range(num_simulations):
        u_array = np.random.uniform(0, 1, trading_days)
        path_returns = np.zeros(trading_days)

        left_mask = u_array < (tail_percentile / 100)
        if np.any(left_mask):
            tail_draws = genpareto.rvs(
                l_xi, loc=0, scale=l_beta, size=np.sum(left_mask)
            )
            path_returns[left_mask] = (l_thresh - historical_mean) - tail_draws

        right_mask = u_array > (1 - (tail_percentile / 100))
        if np.any(right_mask):
            tail_draws = genpareto.rvs(
                r_xi, loc=0, scale=r_beta, size=np.sum(right_mask)
            )
            path_returns[right_mask] = (r_thresh - historical_mean) + tail_draws

        center_mask = ~(left_mask | right_mask)
        if np.any(center_mask):
            path_returns[center_mask] = np.random.choice(
                center_returns, size=np.sum(center_mask)
            )

        drift_adjustment = (
            mu - q - 0.5 * float(np.var(hist_returns))
        ) * trading_days * dt
        S_T[i] = S_0 * np.exp(np.sum(path_returns) + drift_adjustment)

    payoffs = calculate_payoff(S_T, legs)
    expected_value = float(np.mean(payoffs) * np.exp(-r * T))

    return expected_value, S_T, payoffs, float(l_xi), float(r_xi)


# ==========================================
# GARCH-EVT ENGINE (MODEL 4)
# ==========================================


def fit_garch_11(returns: np.ndarray) -> tuple:
    centered_returns = returns - np.mean(returns)
    var_long_term = np.var(centered_returns)

    def garch_variance(params, data):
        omega, alpha, beta = params
        var_t = np.zeros(len(data))
        var_t[0] = var_long_term
        for i in range(1, len(data)):
            var_t[i] = omega + alpha * data[i - 1] ** 2 + beta * var_t[i - 1]
        return var_t

    def garch_log_likelihood(params, data):
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or (alpha + beta) >= 1:
            return 1e10

        var_t = garch_variance(params, data)
        var_t = np.maximum(var_t, 1e-10)

        nll = np.sum(np.log(var_t) + (data**2) / var_t)
        return nll

    initial_guess = [var_long_term * 0.05, 0.1, 0.85]
    bounds = ((1e-8, None), (1e-6, 0.99), (1e-6, 0.99))

    result = minimize(
        garch_log_likelihood,
        initial_guess,
        args=(centered_returns,),
        bounds=bounds,
        method="L-BFGS-B",
    )

    omega, alpha, beta = result.x
    hist_var_array = garch_variance(result.x, centered_returns)

    return omega, alpha, beta, hist_var_array


def extract_garch_innovations(
    returns: np.ndarray, hist_var_array: np.ndarray
) -> np.ndarray:
    centered_returns = returns - np.mean(returns)
    innovations = centered_returns / np.sqrt(hist_var_array)
    return innovations


def value_option_garch_evt(
    S_0: float,
    legs: list,
    T: float,
    r: float,
    mu: float,
    hist_returns: np.ndarray,
    q: float = 0.0,
    tail_percentile: float = 5,
    num_simulations: int = 10000,
    trading_days: int = 30,
) -> tuple:
    dt = 1 / 252

    omega, alpha, beta, hist_var = fit_garch_11(hist_returns)
    innovations = extract_garch_innovations(hist_returns, hist_var)

    current_var_0 = hist_var[-1]

    l_thresh, l_xi, l_beta = fit_gpd(
        innovations, percentile=tail_percentile, tail="left"
    )
    r_thresh, r_xi, r_beta = fit_gpd(
        innovations, percentile=tail_percentile, tail="right"
    )

    center_innovations = innovations[
        (innovations >= l_thresh) & (innovations <= r_thresh)
    ]
    if len(center_innovations) == 0:
        center_innovations = innovations

    S_T = np.zeros(num_simulations)

    for i in range(num_simulations):
        u_array = np.random.uniform(0, 1, trading_days)
        z_path = np.zeros(trading_days)

        left_mask = u_array < (tail_percentile / 100)
        if np.any(left_mask):
            tail_draws = genpareto.rvs(
                l_xi, loc=0, scale=l_beta, size=np.sum(left_mask)
            )
            z_path[left_mask] = l_thresh - tail_draws

        right_mask = u_array > (1 - (tail_percentile / 100))
        if np.any(right_mask):
            tail_draws = genpareto.rvs(
                r_xi, loc=0, scale=r_beta, size=np.sum(right_mask)
            )
            z_path[right_mask] = r_thresh + tail_draws

        center_mask = ~(left_mask | right_mask)
        if np.any(center_mask):
            z_path[center_mask] = np.random.choice(
                center_innovations, size=np.sum(center_mask)
            )

        current_S = S_0
        current_var = current_var_0

        for t in range(trading_days):
            z_t = z_path[t]

            drift_t = (mu - q - 0.5 * (current_var * 252)) * dt
            return_t = drift_t + np.sqrt(current_var) * z_t
            current_S = current_S * np.exp(return_t)

            shock_squared = (np.sqrt(current_var) * z_t) ** 2
            current_var = omega + alpha * shock_squared + beta * current_var

        S_T[i] = current_S

    payoffs = calculate_payoff(S_T, legs)
    expected_value = float(np.mean(payoffs) * np.exp(-r * T))

    return expected_value, S_T, payoffs, omega, alpha, beta


# ==========================================
# UTILITY & PLOTTING FUNCTIONS
# ==========================================


def safe_pct_diff(val: float, base: float) -> str:
    try:
        v = float(val)
        b = float(base)
        if b != 0:
            return f"{((v/b)-1)*100:.1f}% vs BS"
    except Exception:
        pass
    return ""


def generate_percentile_string(S_T_array: np.ndarray, strikes: list) -> str:
    parts = []
    for k in strikes:
        pct = np.mean(S_T_array <= k) * 100
        parts.append(f"**${k:,.2f}**: {pct:.1f}th Percentile")
    return " | ".join(parts)


def plot_structure_payoff(
    S_0: float, legs: list, structure_name: str, direction: str
) -> go.Figure:
    strikes = sorted(list({leg["strike"] for leg in legs}))
    min_K = min(strikes + [S_0])
    max_K = max(strikes + [S_0])

    S_range = np.linspace(min_K * 0.7, max_K * 1.3, 500)
    payoffs = calculate_payoff(S_range, legs)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=payoffs,
            mode="lines",
            name="Terminal Cash Flow",
            line={"color": "#265d74", "width": 3},
        )
    )

    fig.add_vline(
        x=S_0,
        line_dash="dash",
        line_color="#c81d25",
        annotation_text="Spot Price",
    )
    fig.add_hline(y=0, line_dash="solid", line_color="black", opacity=0.3)

    for K in strikes:
        pct_diff = ((K / S_0) - 1) * 100
        sign = "+" if pct_diff > 0 else ""
        payoff_at_K = float(calculate_payoff(np.array([K]), legs)[0])

        fig.add_vline(x=K, line_dash="dot", line_color="#888888", opacity=0.5)
        fig.add_annotation(
            x=K,
            y=payoff_at_K,
            text=f"<b>K: {K}</b><br>{sign}{pct_diff:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#666666",
            ax=0,
            ay=-45,
            font={"size": 11},
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=4,
        )

    fig.update_layout(
        title=f"Terminal Payoff Profile: {direction} {structure_name}",
        xaxis_title="Underlying Price at Expiration ($)",
        yaxis_title="Intrinsic Value / Terminal Cash Flow ($)",
        template="plotly_white",
        height=380,
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_histograms(
    S_T: np.ndarray,
    payoffs: np.ndarray,
    title_prefix: str,
    structure_cost: float,
    xrange_ST: list = None,
    xrange_payoff: list = None,
    xbins_ST: dict = None,
    xbins_payoff: dict = None,
    ymax_ST: float = None,
    ymax_payoff: float = None,
    vol_dist: np.ndarray = None,
    bs_vol_marker: float = None,
    rolling_window_days: int = None,
) -> go.Figure:
    if vol_dist is not None:
        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=(
                "Underlying Price at Expiration (S_T)",
                "Option Structure Value at Expiration",
                "Sampling Volatility Distribution",
            ),
            horizontal_spacing=0.08,
        )
    else:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                "Underlying Price at Expiration (S_T)",
                "Option Structure Value at Expiration",
            ),
        )

    fig.add_trace(
        go.Histogram(
            x=S_T,
            name="S_T",
            marker_color="#265d74",
            opacity=0.75,
            histnorm="percent",
            xbins=xbins_ST if xbins_ST else None,
            autobinx=not xbins_ST,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Histogram(
            x=payoffs,
            name="Value",
            marker_color="#265d74",
            opacity=0.75,
            histnorm="percent",
            xbins=xbins_payoff if xbins_payoff else None,
            autobinx=not xbins_payoff,
        ),
        row=1,
        col=2,
    )

    fig.add_vline(
        x=structure_cost,
        line_dash="dash",
        line_color="#c81d25",
        row=1,
        col=2,
        annotation_text=f"Expected: ${structure_cost:.2f}",
    )

    if vol_dist is not None:
        fig.add_trace(
            go.Histogram(
                x=vol_dist,
                name="Vol Dist",
                marker_color="#265d74",
                opacity=0.75,
                histnorm="percent",
            ),
            row=1,
            col=3,
        )
        if bs_vol_marker is not None:
            fig.add_vline(
                x=bs_vol_marker,
                line_dash="dash",
                line_color="#c81d25",
                row=1,
                col=3,
                annotation_text=f"Full History Vol: {bs_vol_marker*100:.1f}%",
            )

        x_axis_title = (
            f"{rolling_window_days}-Day Rolling Volatility"
            if rolling_window_days
            else "Volatility"
        )
        fig.update_xaxes(
            title_text=x_axis_title, tickformat=".1%", row=1, col=3
        )
        fig.update_yaxes(
            title_text="Relative Frequency (%)", showgrid=False, row=1, col=3
        )

    fig.update_layout(
        height=400,
        showlegend=False,
        template="plotly_white",
        margin={"l": 20, "r": 20, "t": 40, "b": 20},
    )
    fig.update_xaxes(title_text="Price ($)", row=1, col=1)
    fig.update_xaxes(title_text="Value ($)", row=1, col=2)
    fig.update_yaxes(
        title_text="Relative Frequency (%)", showgrid=False, row=1, col=1
    )
    fig.update_yaxes(
        title_text="Relative Frequency (%)", showgrid=False, row=1, col=2
    )

    if xrange_ST is not None:
        fig.update_xaxes(range=xrange_ST, row=1, col=1)
    if xrange_payoff is not None:
        fig.update_xaxes(range=xrange_payoff, row=1, col=2)
    if ymax_ST is not None:
        fig.update_yaxes(range=[0, ymax_ST], row=1, col=1)
    if ymax_payoff is not None:
        fig.update_yaxes(range=[0, ymax_payoff], row=1, col=2)

    return fig


# --- GREEK PROFILER PLOTTING FUNCTIONS ---


def plot_t_step_profile(S_range, S_0, strikes, legs, T, r, q, sigma, greek):
    fig = go.Figure()

    val_0 = calculate_structure_greek(S_range, legs, T, r, q, sigma, greek)
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=val_0,
            mode="lines",
            name=f"Current (T={int(T*365)} Days)",
            line={"color": "#265d74", "width": 3},
        )
    )

    T_half = max(T / 2.0, 1e-4)
    val_half = calculate_structure_greek(
        S_range, legs, T_half, r, q, sigma, greek
    )
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=val_half,
            mode="lines",
            name=f"Halfway (T={int(T_half*365)} Days)",
            line={"color": "#d98880", "width": 2, "dash": "dash"},
        )
    )

    T_exp = 1.0 / 365.0
    val_exp = calculate_structure_greek(
        S_range, legs, T_exp, r, q, sigma, greek
    )
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=val_exp,
            mode="lines",
            name="Expiry (T=1 Day)",
            line={"color": "#c81d25", "width": 2, "dash": "dot"},
        )
    )

    for K in strikes:
        pct_diff = ((K / S_0) - 1) * 100
        sign = "+" if pct_diff > 0 else ""
        val_at_K = float(
            calculate_structure_greek(
                np.array([K]), legs, T, r, q, sigma, greek
            )[0]
        )

        fig.add_vline(x=K, line_dash="dot", line_color="#888888", opacity=0.5)
        fig.add_annotation(
            x=K,
            y=val_at_K,
            text=f"<b>K: {K}</b><br>{sign}{pct_diff:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#666666",
            ax=0,
            ay=-45,
            font={"size": 11},
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=4,
        )

    fig.add_vline(
        x=S_0, line_dash="dash", line_color="black", annotation_text="Spot"
    )
    fig.update_layout(
        title=f"{greek} vs Spot Price (Time Decay Profiler)",
        xaxis_title="Underlying Price ($)",
        yaxis_title=f"Total Structure {greek}",
        template="plotly_white",
        height=450,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_vol_shock_profile(
    S_range, S_0, strikes, legs, T, r, q, sigma_base, rolling_vol, greek
):
    """Updated to use sigma_base for the base curve, while using rolling_vol for the crush (10th pct) and shock (99th pct) limits."""
    fig = go.Figure()

    shock_down = np.percentile(rolling_vol, 10)
    shock_up = np.percentile(rolling_vol, 99)

    # Base Trace evaluates at the Analytical Full History Baseline (bs_sigma)
    val_base = calculate_structure_greek(
        S_range, legs, T, r, q, sigma_base, greek
    )
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=val_base,
            mode="lines",
            name=f"Base Vol: Avg ({sigma_base*100:.1f}%)",
            line={"color": "#265d74", "width": 3},
        )
    )

    val_up = calculate_structure_greek(S_range, legs, T, r, q, shock_up, greek)
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=val_up,
            mode="lines",
            name=f"Vol Shock: 99th Pct ({shock_up*100:.1f}%)",
            line={"color": "#c81d25", "width": 2, "dash": "dash"},
        )
    )

    val_down = calculate_structure_greek(
        S_range, legs, T, r, q, shock_down, greek
    )
    fig.add_trace(
        go.Scatter(
            x=S_range,
            y=val_down,
            mode="lines",
            name=f"Vol Crush: 10th Pct ({shock_down*100:.1f}%)",
            line={"color": "#7dcea0", "width": 2, "dash": "dot"},
        )
    )

    for K in strikes:
        pct_diff = ((K / S_0) - 1) * 100
        sign = "+" if pct_diff > 0 else ""
        val_at_K = float(
            calculate_structure_greek(
                np.array([K]), legs, T, r, q, sigma_base, greek
            )[0]
        )

        fig.add_vline(x=K, line_dash="dot", line_color="#888888", opacity=0.5)
        fig.add_annotation(
            x=K,
            y=val_at_K,
            text=f"<b>K: {K}</b><br>{sign}{pct_diff:.1f}%",
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor="#666666",
            ax=0,
            ay=-45,
            font={"size": 11},
            bgcolor="rgba(255, 255, 255, 0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=4,
        )

    fig.add_vline(
        x=S_0, line_dash="dash", line_color="black", annotation_text="Spot"
    )
    fig.update_layout(
        title=f"{greek} vs Spot Price (Volatility Shock Sensitivity)",
        xaxis_title="Underlying Price ($)",
        yaxis_title=f"Total Structure {greek}",
        template="plotly_white",
        height=450,
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_3d_risk_surface_time(
    S_range, S_0, strikes, legs, T, r, q, sigma, greek
):
    days = np.linspace(int(T * 365), 1, 30)
    T_array = days / 365.0

    Z = np.zeros((len(T_array), len(S_range)))
    for i, t_val in enumerate(T_array):
        Z[i, :] = calculate_structure_greek(
            S_range, legs, t_val, r, q, sigma, greek
        )

    fig = go.Figure(
        data=[go.Surface(z=Z, x=S_range, y=days, colorscale="Viridis")]
    )

    scene_annotations = []
    for K in strikes:
        pct_diff = ((K / S_0) - 1) * 100
        sign = "+" if pct_diff > 0 else ""
        z_val = float(
            calculate_structure_greek(
                np.array([K]), legs, T_array[0], r, q, sigma, greek
            )[0]
        )
        scene_annotations.append(
            {
                "x": K,
                "y": days[0],
                "z": z_val,
                "text": f"<b>K: {K}</b><br>{sign}{pct_diff:.1f}%",
                "showarrow": True,
                "arrowhead": 2,
                "ax": 0,
                "ay": -40,
                "font": {"size": 10},
                "bgcolor": "rgba(255, 255, 255, 0.9)",
                "bordercolor": "#cccccc",
                "borderwidth": 1,
            }
        )

    fig.update_layout(
        title=f"3D Surface: {greek} vs Spot vs Time",
        scene={
            "xaxis_title": "Spot Price ($)",
            "yaxis_title": "Days to Expiry",
            "zaxis_title": greek,
            "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 1.2}},
            "annotations": scene_annotations,
        },
        height=600,
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


def plot_3d_risk_surface_vol(
    S_range, S_0, strikes, legs, T, r, q, sigma_base, greek
):
    vols = np.linspace(max(0.05, sigma_base - 0.20), sigma_base + 0.30, 30)

    Z = np.zeros((len(vols), len(S_range)))
    for i, v_val in enumerate(vols):
        Z[i, :] = calculate_structure_greek(
            S_range, legs, T, r, q, v_val, greek
        )

    fig = go.Figure(
        data=[go.Surface(z=Z, x=S_range, y=vols * 100, colorscale="Plasma")]
    )

    scene_annotations = []
    for K in strikes:
        pct_diff = ((K / S_0) - 1) * 100
        sign = "+" if pct_diff > 0 else ""
        z_val = float(
            calculate_structure_greek(
                np.array([K]), legs, T, r, q, vols[0], greek
            )[0]
        )
        scene_annotations.append(
            {
                "x": K,
                "y": vols[0] * 100,
                "z": z_val,
                "text": f"<b>K: {K}</b><br>{sign}{pct_diff:.1f}%",
                "showarrow": True,
                "arrowhead": 2,
                "ax": 0,
                "ay": -40,
                "font": {"size": 10},
                "bgcolor": "rgba(255, 255, 255, 0.9)",
                "bordercolor": "#cccccc",
                "borderwidth": 1,
            }
        )

    fig.update_layout(
        title=f"3D Surface: {greek} vs Spot vs Volatility",
        scene={
            "xaxis_title": "Spot Price ($)",
            "yaxis_title": "Implied Vol (%)",
            "zaxis_title": greek,
            "camera": {"eye": {"x": 1.5, "y": -1.5, "z": 1.2}},
            "annotations": scene_annotations,
        },
        height=600,
        margin={"l": 0, "r": 0, "b": 0, "t": 40},
    )
    return fig


# ==========================================
# TOOLTIP EXPLANATIONS (MARKDOWN & LATEX)
# ==========================================

help_text_m1 = r"""
**The Intuition:** Values the option analytically assuming volatility is perfectly constant and returns follow a smooth Lognormal distribution (no Black Swans).

**The Math (Distributions & Drift):**
* **Adjusted Drift:** $\mu_{adj} = \mu - q - \frac{1}{2}\sigma^2$
* **1-Period Return:** $\ln\left(\frac{S_{t+\Delta t}}{S_t}\right) \sim N(\mu_{adj} \Delta t, \sigma^2 \Delta t)$
* **Terminal Price:** $S_T \sim \text{Lognormal}(\ln S_0 + \mu_{adj} T, \sigma^2 T)$
* **Terminal Price ($Z$):** $S_T = S_0 \exp(\mu_{adj} T + \sigma \sqrt{T} Z)$ where $Z \sim N(0,1)$

**Implementation:** Uses the closed-form analytical Black-Scholes formula, substituting the risk-free rate ($r$) with the subjective physical drift ($\mu$) inside the probabilities.
"""

help_text_m2 = r"""
**The Intuition:** Recognizes volatility clustering. Instead of assuming one constant volatility, it repeatedly samples from historically realized N-day volatility regimes.

**The Math (Distributions & Drift):**
* **Adjusted Drift for Sample $i$:** $\mu_{adj, i} = \mu - q - \frac{1}{2}\sigma_i^2$
* **Terminal Price ($Z$):** $S_T = S_0 \exp(\mu_{adj, i} T + \sigma_i \sqrt{T} Z)$ where $Z \sim N(0,1)$

**Implementation:** Monte Carlo simulation. The engine draws a historical volatility $\sigma_i$ at random, evaluates the analytical Black-Scholes formula using that specific volatility, repeats 10,000 times, and averages the outcomes to create a fat-tailed "Mixture Distribution."

**Why This Model Prices Higher (Jensen's Inequality):**
Option prices are strictly convex to volatility (positive Volga/Vomma). A massive volatility spike adds more dollar value to the option than a massive volatility drop takes away. According to **Jensen's Inequality** ($E[f(X)] \ge f(E[X])$), the average of the option prices across varied volatilities (Model 2) will always be greater than the option price calculated using a single average volatility (Model 1).
"""

help_text_m3 = r"""
**The Intuition:** Abandons Normal distribution assumptions entirely. Simulates daily steps using exact historical returns, augmented with Extreme Value Theory (EVT) to accurately model physical "Black Swan" jumps.

**The Math (Distributions & Drift):**
* **Adjusted Drift:** $\mu_{adj} = \mu - q - \frac{1}{2}\sigma^2$
* **1-Period Return:** $R_t = \ln\left(\frac{S_t}{S_{t-1}}\right) \sim F_{EVT}$ (Empirical Center + GPD Tails)
* **Terminal Price:** $S_T = S_0 \exp\left( \sum_{t=1}^{N} R_t + \mu_{adj} T \right)$

**Implementation:** Simulates the path day-by-day. It draws a uniform random variable $U \sim U(0,1)$. If $U < 0.05$, it draws a massive crash from the Left GPD tail. If $U > 0.95$, it draws a squeeze from the Right GPD tail. Otherwise, it samples a normal historical day.
"""

help_text_m4 = r"""
**The Intuition:** The gold standard for path dependency. Recognizes that an extreme shock today massively spikes the baseline volatility tomorrow (GARCH), creating sustained market panic.

**The Math (Distributions & Drift):**
* **1-Period Return:** $R_t = \mu_t + \sigma_{t-1} Z_t$
* **The Shock (Innovation):** $Z_t \sim F_{EVT}$ (EVT fitted specifically to standardized shocks)
* **Dynamic Variance:** $\sigma_t^2 = \omega + \alpha (R_{t-1} - \mu_{t-1})^2 + \beta \sigma_{t-1}^2$
* **Dynamic Drift:** $\mu_t = (\mu - q - \frac{1}{2}\sigma_{t-1}^2)\Delta t$
* **Terminal Price:** $S_T = S_0 \exp\left( \sum_{t=1}^{N} R_t \right)$

**Implementation:** Starts with today's volatility. Draws an EVT shock ($Z_t$), calculates the physical return ($R_t$), and critically, feeds that shock into the GARCH equation to update tomorrow's volatility ($\sigma_t$). A massive shock early in the simulation forces the stock to thrash violently for the remaining days.
"""

help_text_left_tail = r"""
**Left Tail Shape Parameter ($\xi$)**
This quantifies the "fatness" of the negative tail (market crashes). 

* **$\xi \le 0$:** Thin tails (Normal risk).
* **$0 < \xi < 0.5$:** Heavy tails. Crashes are far more likely than a Normal distribution assumes.
* **$\xi \ge 0.5$:** Infinite Variance. The tail is so fat that historical worst-case scenarios do not limit future worst-case scenarios. 

*A higher number means put options should carry a massive premium.*
"""

help_text_right_tail = r"""
**Right Tail Shape Parameter ($\xi$)**
This quantifies the "fatness" of the positive tail (market squeezes and euphoric rallies). 

Because markets typically fall faster than they rise, this number is usually lower than the Left Tail index. If this number spikes, it indicates severe upside tail risk (e.g., a meme-stock short squeeze regime), increasing the fair value of deep out-of-the-money calls.
"""

# Native Header Tooltip for Option Greeks Profile Section
help_text_greeks = r"""
### **Option Greeks Summary Guide**

**First-Order Greeks (Direct Sensitivities)**
* **Delta ($\Delta$):** $\frac{\partial V}{\partial S}$ — Directional risk. Expected change in option value per $\$1.00$ change in the underlying asset price. Also acts as a proxy for In-The-Money probability.
* **Vega ($\nu$):** $\frac{\partial V}{\partial \sigma}$ — Volatility risk. Expected change in option value per $1\%$ absolute change in implied volatility. Long options are always long Vega.
* **Theta ($\Theta$):** $\frac{\partial V}{\partial t}$ — Time decay. Expected daily dollar loss in option value strictly due to the passage of time. Accelerates toward expiration.
* **Rho ($\rho$):** $\frac{\partial V}{\partial r}$ — Interest rate risk. Sensitivity to a $1\%$ change in the risk-free interest rate.

**Second-Order Greeks (Convexity & Acceleration)**
* **Gamma ($\Gamma$):** $\frac{\partial^2 V}{\partial S^2}$ or $\frac{\partial \Delta}{\partial S}$ — Directional acceleration. Rate of change in Delta per $\$1.00$ change in stock price. Highly concentrated At-The-Money.
* **Vomma / Volga:** $\frac{\partial^2 V}{\partial \sigma^2}$ or $\frac{\partial \nu}{\partial \sigma}$ — Volatility convexity. Rate of change in Vega per $1\%$ change in implied volatility. Captures the upward curve of option premium under extreme fear.
* **Vanna:** $\frac{\partial^2 V}{\partial S \partial \sigma}$ — Cross-derivative. Rate of change in Delta per $1\%$ change in volatility (or Vega per $\$1.00$ spot move). Shows how volatility spikes drag Out-Of-The-Money options closer to the money.
* **Charm:** $\frac{\partial^2 V}{\partial S \partial t}$ — Delta decay. The absolute daily change in Delta due to the passage of time. Magnetizes OTM Delta to 0 and ITM Delta to 1.
"""


# ==========================================
# 2. STREAMLIT APP UI & LOGIC
# ==========================================

st.set_page_config(page_title="Option Valuation Tool", layout="wide")

st.markdown(
    """
<style>
div.stButton > button[kind="primary"] {
    background-color: #265d74 !important;
    border-color: #265d74 !important;
}
div.stButton > button[kind="primary"]:hover {
    background-color: #1a4253 !important;
    border-color: #1a4253 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Option Valuation Tool")

st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
st.sidebar.title("Input Parameters")

st.sidebar.header("1. Asset & Expiry")
ticker = st.sidebar.text_input("Underlying Ticker (e.g., SPY)", value="SPY").upper()

today = datetime.date.today()
default_expiry = today + datetime.timedelta(days=30)
expiry_date = st.sidebar.date_input(
    "Expiration Date",
    value=default_expiry,
    min_value=today + datetime.timedelta(days=1),
)

cal = USTradingCalendar()
holidays = cal.holidays(start=today, end=expiry_date).values.astype("datetime64[D]")

days_to_expiry = (expiry_date - today).days
T = days_to_expiry / 365.0
trading_days_to_expiry = int(
    max(1, np.busday_count(today, expiry_date, holidays=holidays))
)

st.sidebar.markdown(
    f"""
<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px;">
    <b>🗓️ Calendar Days:</b> {days_to_expiry}<br>
    <b>📈 Trading Days:</b> {trading_days_to_expiry}
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("2. Option Structure")
STRUCTURES = [
    "Call",
    "Put",
    "Call Spread",
    "Put Spread",
    "Put Spread + Call",
    "Call Spread + Put",
    "Butterfly",
    "Iron Butterfly",
    "Condor",
    "Iron Condor",
    "Straddle",
    "Strangle",
    "Synthetic",
]
structure = st.sidebar.selectbox("Select Strategy", STRUCTURES)

direction = st.sidebar.selectbox("Direction", ["Long", "Short"])
dir_mult = 1 if direction == "Long" else -1


def act(pos):
    net = pos * dir_mult
    return "Buy" if net > 0 else "Sell"


st.sidebar.markdown("**Input Strikes (K1 to K4 from Lowest to Highest)**")

K1, K2, K3, K4 = 0, 0, 0, 0

if structure in ["Call", "Put", "Straddle"]:
    K1 = st.sidebar.number_input(f"Strike (K1) - [{act(1)}]", value=500.0, step=1.0)
elif structure == "Synthetic":
    K1 = st.sidebar.number_input(
        f"Strike (K1) - [Call: {act(1)} | Put: {act(-1)}]", value=500.0, step=1.0
    )
elif structure == "Call Spread":
    K1 = st.sidebar.number_input(
        f"Lower Strike Call (K1) - [{act(1)}]", value=490.0, step=1.0
    )
    K2 = st.sidebar.number_input(
        f"Upper Strike Call (K2) - [{act(-1)}]", value=510.0, step=1.0
    )
elif structure == "Put Spread":
    K1 = st.sidebar.number_input(
        f"Lower Strike Put (K1) - [{act(-1)}]", value=490.0, step=1.0
    )
    K2 = st.sidebar.number_input(
        f"Upper Strike Put (K2) - [{act(1)}]", value=510.0, step=1.0
    )
elif structure == "Put Spread + Call":
    K1 = st.sidebar.number_input(
        f"Lower Strike Put (K1) - [{act(-1)}]", value=480.0, step=1.0
    )
    K2 = st.sidebar.number_input(
        f"Middle Strike Put (K2) - [{act(1)}]", value=500.0, step=1.0
    )
    K3 = st.sidebar.number_input(
        f"Upper Strike Call (K3) - [{act(1)}]", value=520.0, step=1.0
    )
elif structure == "Call Spread + Put":
    K1 = st.sidebar.number_input(
        f"Lower Strike Call (K1) - [{act(1)}]", value=480.0, step=1.0
    )
    K2 = st.sidebar.number_input(
        f"Middle Strike Call (K2) - [{act(-1)}]", value=500.0, step=1.0
    )
    K3 = st.sidebar.number_input(
        f"Upper Strike Put (K3) - [{act(1)}]", value=520.0, step=1.0
    )
elif structure in ["Butterfly", "Iron Butterfly"]:
    K1 = st.sidebar.number_input(
        f"Lower Strike (K1) - [{act(1)}]", value=480.0, step=1.0
    )
    K2 = st.sidebar.number_input(
        f"Middle Strike (K2) - [{act(-1)}]", value=500.0, step=1.0
    )
    K3 = st.sidebar.number_input(
        f"Upper Strike (K3) - [{act(1)}]", value=520.0, step=1.0
    )
elif structure in ["Condor", "Iron Condor"]:
    K1 = st.sidebar.number_input(
        f"Lowest Strike (K1) - [{act(1)}]", value=470.0, step=1.0
    )
    K2 = st.sidebar.number_input(
        f"Lower Middle Strike (K2) - [{act(-1)}]", value=490.0, step=1.0
    )
    K3 = st.sidebar.number_input(
        f"Upper Middle Strike (K3) - [{act(-1)}]", value=510.0, step=1.0
    )
    K4 = st.sidebar.number_input(
        f"Highest Strike (K4) - [{act(1)}]", value=530.0, step=1.0
    )
elif structure == "Strangle":
    K1 = st.sidebar.number_input(
        f"Lower Strike Put (K1) - [{act(1)}]", value=490.0, step=1.0
    )
    K2 = st.sidebar.number_input(
        f"Upper Strike Call (K2) - [{act(1)}]", value=510.0, step=1.0
    )

legs = []
if structure == "Call":
    legs = [{"type": "call", "strike": K1, "pos": 1}]
elif structure == "Put":
    legs = [{"type": "put", "strike": K1, "pos": 1}]
elif structure == "Call Spread":
    legs = [
        {"type": "call", "strike": K1, "pos": 1},
        {"type": "call", "strike": K2, "pos": -1},
    ]
elif structure == "Put Spread":
    legs = [
        {"type": "put", "strike": K2, "pos": 1},
        {"type": "put", "strike": K1, "pos": -1},
    ]
elif structure == "Put Spread + Call":
    legs = [
        {"type": "put", "strike": K2, "pos": 1},
        {"type": "put", "strike": K1, "pos": -1},
        {"type": "call", "strike": K3, "pos": 1},
    ]
elif structure == "Call Spread + Put":
    legs = [
        {"type": "call", "strike": K1, "pos": 1},
        {"type": "call", "strike": K2, "pos": -1},
        {"type": "put", "strike": K3, "pos": 1},
    ]
elif structure == "Butterfly":
    legs = [
        {"type": "call", "strike": K1, "pos": 1},
        {"type": "call", "strike": K2, "pos": -2},
        {"type": "call", "strike": K3, "pos": 1},
    ]
elif structure == "Iron Butterfly":
    legs = [
        {"type": "put", "strike": K1, "pos": 1},
        {"type": "put", "strike": K2, "pos": -1},
        {"type": "call", "strike": K2, "pos": -1},
        {"type": "call", "strike": K3, "pos": 1},
    ]
elif structure == "Condor":
    legs = [
        {"type": "call", "strike": K1, "pos": 1},
        {"type": "call", "strike": K2, "pos": -1},
        {"type": "call", "strike": K3, "pos": -1},
        {"type": "call", "strike": K4, "pos": 1},
    ]
elif structure == "Iron Condor":
    legs = [
        {"type": "put", "strike": K1, "pos": 1},
        {"type": "put", "strike": K2, "pos": -1},
        {"type": "call", "strike": K3, "pos": -1},
        {"type": "call", "strike": K4, "pos": 1},
    ]
elif structure == "Straddle":
    legs = [
        {"type": "call", "strike": K1, "pos": 1},
        {"type": "put", "strike": K1, "pos": 1},
    ]
elif structure == "Strangle":
    legs = [
        {"type": "put", "strike": K1, "pos": 1},
        {"type": "call", "strike": K2, "pos": 1},
    ]
elif structure == "Synthetic":
    legs = [
        {"type": "call", "strike": K1, "pos": 1},
        {"type": "put", "strike": K1, "pos": -1},
    ]

for leg in legs:
    leg["pos"] *= dir_mult

st.sidebar.header("3. Market Dynamics")
r = st.sidebar.number_input(
    "Risk-Free Rate (r)", value=0.036, step=0.001, format="%.3f"
)
q = st.sidebar.number_input(
    "Dividend Yield (q)", value=0.014, step=0.001, format="%.3f"
)

st.sidebar.markdown("**Expected Drift (\u03bc) Components**")
beta = st.sidebar.number_input("Stock Beta (\u03b2)", value=1.000, step=0.001, format="%.3f")
alpha = st.sidebar.number_input(
    "Alpha (\u03b1)",
    value=0.000,
    step=0.001,
    format="%.3f",
    help="Expected convergence: (Fair Value / Spot) - 1",
)
erp = st.sidebar.number_input(
    "Equity Risk Premium (ERP)", value=0.050, step=0.001, format="%.3f"
)

mu = r + (beta * erp) + (alpha / T)

st.sidebar.markdown(
    f"""
<div style="background-color: #e8f4f8; padding: 10px; border-radius: 5px; margin-top: 10px;">
    <b>Calculated Expected Drift (\u03bc):</b> {mu*100:.3f}%
</div>
<br>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("4. Simulation Parameters")
sim_options = [10000, 50000, 100000, 250000, 500000, 1000000]
num_simulations = st.sidebar.selectbox("Number of Simulations", sim_options, index=0)

if "run_sim" not in st.session_state:
    st.session_state.run_sim = False

if st.sidebar.button("Fetch Data & Value Options", type="primary"):
    st.session_state.run_sim = True

if st.session_state.run_sim:
    with st.spinner(f"Fetching max historical data for {ticker}..."):
        try:
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(period="max")

            if data.empty:
                st.error("No data found for this ticker.")
                st.stop()

            start_date = data.index.min().strftime("%Y-%m-%d")
            end_date = data.index.max().strftime("%Y-%m-%d")
            total_days = len(data)

            close_prices = data["Close"]
            if isinstance(close_prices, pd.DataFrame):
                close_prices = close_prices.iloc[:, 0]

            close_array = close_prices.to_numpy(dtype=float).flatten()
            S_0 = float(close_array[-1])

            returns = np.log(close_array[1:] / close_array[:-1])
            returns = returns[~np.isnan(returns)]

            bs_sigma = float(np.std(returns) * np.sqrt(252))
            years_of_data = len(returns) / 252.0
            bs_sigma_label = f"Full Data ({years_of_data:.1f} Years)"

            if len(returns) < trading_days_to_expiry:
                st.error(
                    f"Not enough historical data to calculate a {trading_days_to_expiry}-trading-day rolling window."
                )
                st.stop()

            rolling_vol = np.array(
                [
                    np.std(returns[i - trading_days_to_expiry : i]) * np.sqrt(252)
                    for i in range(trading_days_to_expiry, len(returns))
                ]
            )
            current_sigma = float(rolling_vol[-1])

            st.sidebar.markdown("---")
            st.sidebar.subheader("📊 Data Description")
            st.sidebar.metric("Spot Price", f"${S_0:.2f}")
            st.sidebar.metric("Historical Vol", f"{bs_sigma*100:.2f}%")
            st.sidebar.metric("Start Date", start_date)
            st.sidebar.metric("End Date", end_date)
            st.sidebar.metric("Trading Years", f"{years_of_data:.1f}")
            st.sidebar.metric("Trading Days", f"{total_days:,}")

        except Exception as e:
            st.error(f"❌ Error during Data Preparation: {e}")
            st.stop()

    st.markdown("---")
    st.subheader("Payout Diagram for Option Structure")

    unique_strikes = sorted(list({leg["strike"] for leg in legs}))

    st.plotly_chart(
        plot_structure_payoff(S_0, legs, structure, direction),
        use_container_width=True,
    )

    with st.expander("🔍 View Model Inputs"):
        st.markdown(
            f"""
        **Static Parameters:**
        * **Spot Price ($S_0$):** ${S_0:.2f}
        * **Expiration Date:** {expiry_date.strftime('%B %d, %Y')} 
        * **Time to Expiry ($T$):** {T:.4f} years ({days_to_expiry} calendar days / {trading_days_to_expiry} trading days)
        * **Expected Drift ($\mu$):** {mu:.3%} *(Calculated as: {r:.3%} + {beta:.3f} $\\times$ {erp:.3%} + {alpha:.3%} / {T:.4f})*
        * **Risk-Free Rate ($r$):** {r:.3%}
        * **Dividend Yield ($q$):** {q:.3%}
        
        **Volatility Inputs by Model:**
        * **BS Sigma ($\sigma$):** {bs_sigma:.3%} *({bs_sigma_label})*
        * **Bootstrap Volatility Array:** {len(rolling_vol)} historical {trading_days_to_expiry}-day rolling samples available for random draw
        
        **Simulation Details:**
        * **Number of Paths (N):** {num_simulations:,}
        
        **Structure Legs:**
        """
        )
        leg_df = pd.DataFrame(legs)
        leg_df.columns = ["Type", "Strike", "Quantity"]
        st.dataframe(leg_df, hide_index=True)

    results = {}
    st_arrays = []
    payoff_arrays = []
    bs_val_baseline = 0.0

    with st.spinner(
        "Calculating analytical baseline and simulating Monte Carlo paths..."
    ):
        try:
            bs_val, bs_ST, bs_payoffs = value_option_black_scholes(
                S_0,
                legs,
                T,
                r,
                mu,
                bs_sigma,
                q,
                num_simulations=num_simulations,
            )
            bs_val_baseline = bs_val
            results["BS"] = (bs_val, bs_ST, bs_payoffs)
            st_arrays.append(bs_ST)
            payoff_arrays.append(bs_payoffs)
        except Exception as e:
            st.error(f"❌ Error in Black-Scholes Calculation: {e}")

        try:
            (
                boot_val,
                boot_ST,
                boot_payoffs,
                boot_sampled_vols,
            ) = value_option_bootstrap_volatility(
                S_0,
                legs,
                T,
                r,
                mu,
                rolling_vol,
                q,
                num_simulations=num_simulations,
            )
            results["Boot"] = (boot_val, boot_ST, boot_payoffs, boot_sampled_vols)
            st_arrays.append(boot_ST)
            payoff_arrays.append(boot_payoffs)
        except Exception as e:
            st.error(f"❌ Error in Bootstrap Volatility Calculation: {e}")

        try:
            (
                evt_val,
                evt_ST,
                evt_payoffs,
                evt_l_xi,
                evt_r_xi,
            ) = value_option_evt_two_tailed(
                S_0,
                legs,
                T,
                r,
                mu,
                hist_returns=returns,
                q=q,
                trading_days=trading_days_to_expiry,
                num_simulations=num_simulations,
            )
            results["EVT"] = (evt_val, evt_ST, evt_payoffs, evt_l_xi, evt_r_xi)
            st_arrays.append(evt_ST)
            payoff_arrays.append(evt_payoffs)
        except Exception as e:
            st.error(f"❌ Error in EVT Calculation: {e}")

        try:
            (
                garch_val,
                garch_ST,
                garch_payoffs,
                g_omega,
                g_alpha,
                g_beta,
            ) = value_option_garch_evt(
                S_0,
                legs,
                T,
                r,
                mu,
                hist_returns=returns,
                q=q,
                trading_days=trading_days_to_expiry,
                num_simulations=num_simulations,
            )
            results["GARCH"] = (
                garch_val,
                garch_ST,
                garch_payoffs,
                g_omega,
                g_alpha,
                g_beta,
            )
            st_arrays.append(garch_ST)
            payoff_arrays.append(garch_payoffs)
        except Exception as e:
            st.error(f"❌ Error in GARCH-EVT Calculation: {e}")

    xrange_ST = None
    xrange_payoff = None
    xbins_ST = None
    xbins_payoff = None
    ymax_ST = None
    ymax_payoff = None

    if st_arrays and payoff_arrays:
        all_ST = np.concatenate(st_arrays)
        all_payoffs = np.concatenate(payoff_arrays)

        st_buffer = (np.max(all_ST) - np.min(all_ST)) * 0.02
        payoff_buffer = (np.max(all_payoffs) - np.min(all_payoffs)) * 0.02

        st_min = np.min(all_ST) - st_buffer
        st_max = np.max(all_ST) + st_buffer
        payoff_min = np.min(all_payoffs) - payoff_buffer
        payoff_max = np.max(all_payoffs) + payoff_buffer

        xrange_ST = [st_min, st_max]
        xrange_payoff = [payoff_min, payoff_max]

        st_size = (st_max - st_min) / 60
        payoff_size = (payoff_max - payoff_min) / 60

        xbins_ST = {"start": st_min, "end": st_max, "size": st_size}
        xbins_payoff = {"start": payoff_min, "end": payoff_max, "size": payoff_size}

        max_freq_ST = 0
        max_freq_payoff = 0

        bins_st_edges = np.linspace(st_min, st_max, 61)
        bins_payoff_edges = np.linspace(payoff_min, payoff_max, 61)

        for st_arr in st_arrays:
            counts, _ = np.histogram(st_arr, bins=bins_st_edges)
            max_freq_ST = max(max_freq_ST, np.max(counts) / len(st_arr) * 100)

        for payoff_arr in payoff_arrays:
            counts, _ = np.histogram(payoff_arr, bins=bins_payoff_edges)
            max_freq_payoff = max(
                max_freq_payoff, np.max(counts) / len(payoff_arr) * 100
            )

        ymax_ST = max_freq_ST * 1.10
        ymax_payoff = max_freq_payoff * 1.10

    if "BS" in results:
        st.subheader(
            "1. Standard Black-Scholes (Analytical Valuation: Full-History Constant Vol)",
            help=help_text_m1,
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Expected Value / Premium", f"${results['BS'][0]:.2f}")

        prob_itm = np.mean(results["BS"][2] > 0) * 100
        m2.metric(
            "Prob. of Expiring ITM",
            f"{prob_itm:.3f}%",
            help="Probability the structure yields a strictly positive intrinsic value (Payoff > $0) at expiration.",
        )

        m3.metric("Constant Volatility Applied", f"{bs_sigma*100:.2f}%")

        st.caption(
            f"📍 **Strike Distribution Mapping:** {generate_percentile_string(results['BS'][1], unique_strikes)}"
        )
        st.plotly_chart(
            plot_histograms(
                results["BS"][1],
                results["BS"][2],
                "Black-Scholes",
                results["BS"][0],
                xrange_ST,
                xrange_payoff,
                xbins_ST,
                xbins_payoff,
                ymax_ST,
                ymax_payoff,
            ),
            use_container_width=True,
        )

    if "Boot" in results:
        st.markdown("---")
        st.subheader(
            "2. Black-Scholes Mixture Model (Monte Carlo Simulation: Bootstrapped Empirical Vol)",
            help=help_text_m2,
        )

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(
            "Expected Value / Premium",
            f"${results['Boot'][0]:.2f}",
            safe_pct_diff(results["Boot"][0], bs_val_baseline),
        )

        prob_itm = np.mean(results["Boot"][2] > 0) * 100
        m2.metric(
            "Prob. of Expiring ITM",
            f"{prob_itm:.3f}%",
            help="Probability the structure yields a strictly positive intrinsic value (Payoff > $0) at expiration.",
        )

        sampled_vols = results["Boot"][3]
        m3.metric("Min Sample Vol", f"{np.min(sampled_vols)*100:.2f}%")
        m4.metric("Median Sample Vol", f"{np.median(sampled_vols)*100:.2f}%")
        m5.metric("Avg Sample Vol", f"{np.mean(sampled_vols)*100:.2f}%")
        m6.metric("Max Sample Vol", f"{np.max(sampled_vols)*100:.2f}%")

        st.caption(
            f"📍 **Strike Distribution Mapping:** {generate_percentile_string(results['Boot'][1], unique_strikes)}"
        )
        st.plotly_chart(
            plot_histograms(
                results["Boot"][1],
                results["Boot"][2],
                "Bootstrap",
                results["Boot"][0],
                xrange_ST,
                xrange_payoff,
                xbins_ST,
                xbins_payoff,
                ymax_ST,
                ymax_payoff,
                vol_dist=rolling_vol,
                bs_vol_marker=bs_sigma,
                rolling_window_days=trading_days_to_expiry,
            ),
            use_container_width=True,
        )

    if "EVT" in results:
        st.markdown("---")
        st.subheader(
            "3. Return-Based EVT Model (Daily Simulation: Empirical Center + EVT Tails)",
            help=help_text_m3,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Expected Value / Premium",
            f"${results['EVT'][0]:.2f}",
            safe_pct_diff(results["EVT"][0], bs_val_baseline),
        )

        prob_itm = np.mean(results["EVT"][2] > 0) * 100
        m2.metric(
            "Prob. of Expiring ITM",
            f"{prob_itm:.3f}%",
            help="Probability the structure yields a strictly positive intrinsic value (Payoff > $0) at expiration.",
        )

        m3.metric(
            "Left Tail Shape (\u03BE)",
            f"{results['EVT'][3]:.4f}",
            help=help_text_left_tail,
        )
        m4.metric(
            "Right Tail Shape (\u03BE)",
            f"{results['EVT'][4]:.4f}",
            help=help_text_right_tail,
        )

        st.caption(
            f"📍 **Strike Distribution Mapping:** {generate_percentile_string(results['EVT'][1], unique_strikes)}"
        )
        st.plotly_chart(
            plot_histograms(
                results["EVT"][1],
                results["EVT"][2],
                "EVT",
                results["EVT"][0],
                xrange_ST,
                xrange_payoff,
                xbins_ST,
                xbins_payoff,
                ymax_ST,
                ymax_payoff,
            ),
            use_container_width=True,
        )

    if "GARCH" in results:
        st.markdown("---")
        st.subheader(
            "4. GARCH-EVT Filtered Historical Sim (Daily Simulation: Dynamic Vol + EVT Innovations)",
            help=help_text_m4,
        )

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric(
            "Expected Value / Premium",
            f"${results['GARCH'][0]:.2f}",
            safe_pct_diff(results["GARCH"][0], bs_val_baseline),
        )

        prob_itm = np.mean(results["GARCH"][2] > 0) * 100
        m2.metric(
            "Prob. of Expiring ITM",
            f"{prob_itm:.3f}%",
            help="Probability the structure yields a strictly positive intrinsic value (Payoff > $0) at expiration.",
        )

        m3.metric(
            "GARCH Omega (\u03C9)",
            f"{results['GARCH'][3]:.6f}",
            help="Baseline long-term variance weight.",
        )
        m4.metric(
            "GARCH Alpha (\u03B1)",
            f"{results['GARCH'][4]:.4f}",
            help="Shock reaction parameter.",
        )
        m5.metric(
            "GARCH Beta (\u03B2)",
            f"{results['GARCH'][5]:.4f}",
            help="Volatility persistence parameter.",
        )
        m6.metric(
            "\u03B1 + \u03B2 (Mean Reversion)",
            f"{(results['GARCH'][4] + results['GARCH'][5]):.4f}",
            help="Must be < 1.0 for mean reversion to exist.",
        )

        st.caption(
            f"📍 **Strike Distribution Mapping:** {generate_percentile_string(results['GARCH'][1], unique_strikes)}"
        )
        st.plotly_chart(
            plot_histograms(
                results["GARCH"][1],
                results["GARCH"][2],
                "GARCH-EVT",
                results["GARCH"][0],
                xrange_ST,
                xrange_payoff,
                xbins_ST,
                xbins_payoff,
                ymax_ST,
                ymax_payoff,
            ),
            use_container_width=True,
        )

    # ==========================================
    # 5. RISK & SENSITIVITY PROFILER
    # ==========================================
    st.markdown("---")

    # Native Header Tooltip Integration via 'help' param
    st.header("5. Risk & Sensitivity Profiler", help=help_text_greeks)

    # Pre-calculate Current State aggregate metrics across Spot array sizing = 1
    spot_arr = np.array([S_0])
    curr_delta = float(
        calculate_structure_greek(spot_arr, legs, T, r, q, bs_sigma, "Delta")[0]
    )
    curr_gamma = float(
        calculate_structure_greek(spot_arr, legs, T, r, q, bs_sigma, "Gamma")[0]
    )
    curr_vanna = float(
        calculate_structure_greek(spot_arr, legs, T, r, q, bs_sigma, "Vanna")[0]
    )
    curr_vega = float(
        calculate_structure_greek(spot_arr, legs, T, r, q, bs_sigma, "Vega")[0]
    )
    curr_vomma = float(
        calculate_structure_greek(spot_arr, legs, T, r, q, bs_sigma, "Vomma")[0]
    )
    curr_theta = float(
        calculate_structure_greek(spot_arr, legs, T, r, q, bs_sigma, "Theta")[0]
    )

    # Render Current Dynamic Metrics Dashboard in logically related order
    st.markdown("#### **Aggregate Structure Greeks at Current State**")
    g1, g2, g3, g4, g5, g6 = st.columns(6)
    g1.metric("Delta (\u0394)", f"{curr_delta:+.3f}")
    g2.metric("Gamma (\u0393)", f"{curr_gamma:.4f}")
    g3.metric("Vanna", f"{curr_vanna:+.4f}")
    g4.metric("Vega (\u03BD)", f"${curr_vega:+.2f}")
    g5.metric("Vomma", f"{curr_vomma:+.4f}")
    g6.metric("Theta (\u0398)", f"${curr_theta:+.2f}")

    st.markdown("<br>", unsafe_allow_html=True)

    min_K_prof = min(unique_strikes + [S_0])
    max_K_prof = max(unique_strikes + [S_0])
    S_range_prof = np.linspace(min_K_prof * 0.7, max_K_prof * 1.3, 300)

    def render_profiler_block(block_number: int, default_index: int, block_title: str):
        st.markdown(f"### {block_title}")

        target_greek = st.selectbox(
            f"Select Target Metric to Profile ({block_number}):",
            ["BS Value", "Delta", "Gamma", "Vega", "Theta", "Rho"],
            index=default_index,
            key=f"prof_select_{block_number}",
        )

        st.markdown(
            f"**Target Metric:** {target_greek} | **Anchor Volatility:** {bs_sigma*100:.1f}%"
        )

        tab1, tab2 = st.tabs(
            [f"2D Scenarios ({target_greek})", f"3D Surfaces ({target_greek})"]
        )

        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_t_step_profile(
                        S_range_prof,
                        S_0,
                        unique_strikes,
                        legs,
                        T,
                        r,
                        q,
                        bs_sigma,
                        target_greek,
                    ),
                    use_container_width=True,
                )
            with col2:
                # Passing both bs_sigma (base) and rolling_vol (shocks) to ensure Premise Alignment
                st.plotly_chart(
                    plot_vol_shock_profile(
                        S_range_prof,
                        S_0,
                        unique_strikes,
                        legs,
                        T,
                        r,
                        q,
                        bs_sigma,
                        rolling_vol,
                        target_greek,
                    ),
                    use_container_width=True,
                )

        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(
                    plot_3d_risk_surface_time(
                        S_range_prof,
                        S_0,
                        unique_strikes,
                        legs,
                        T,
                        r,
                        q,
                        bs_sigma,
                        target_greek,
                    ),
                    use_container_width=True,
                )
            with col2:
                st.plotly_chart(
                    plot_3d_risk_surface_vol(
                        S_range_prof,
                        S_0,
                        unique_strikes,
                        legs,
                        T,
                        r,
                        q,
                        bs_sigma,
                        target_greek,
                    ),
                    use_container_width=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

    render_profiler_block(
        block_number=1,
        default_index=0,
        block_title="Profile 1: Premium & Risk-Neutral Value",
    )
    st.markdown("---")
    render_profiler_block(
        block_number=2,
        default_index=1,
        block_title="Profile 2: First-Order Greeks (Directional Risk)",
    )
    st.markdown("---")
    render_profiler_block(
        block_number=3,
        default_index=3,
        block_title="Profile 3: Second-Order Greeks (Volatility Convexity)",
    )