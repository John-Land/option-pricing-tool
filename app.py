import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import binom, geom, poisson, expon, norm, weibull_min, t, pareto, cauchy, skew, kurtosis, bernoulli

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="LLN & CLT Convergence Simulator", layout="wide")

# --- CSS Styling ---
st.markdown("""
    <style>
    .main-header {
        font-family: 'Inter', sans-serif;
        color: #0f172a;
    }
    .chart-desc {
        font-size: 0.875rem;
        color: #64748b;
        margin-bottom: 1rem;
    }
    .section-header {
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 2px solid #e2e8f0;
        color: #0f172a;
        margin-bottom: 1rem;
    }
    /* Target the text inside the mathematical foundations expander to match st.caption */
    div[data-testid="stExpander"] p, div[data-testid="stExpander"] li, div[data-testid="stExpander"] table {
        font-size: 0.875rem !important;
        color: #64748b;
        line-height: 1.4;
    }
    /* Slightly tighten vertical spacing in the expander */
    div[data-testid="stExpander"] p {
        margin-bottom: 0.5rem;
    }
    div[data-testid="stExpander"] h3 {
        margin-bottom: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>LLN & CLT Convergence Simulator</h1>", unsafe_allow_html=True)

# --- MATHEMATICAL FOUNDATIONS SECTION ---
with st.expander("Mathematical Foundations & Core Concepts", expanded=True):
    col_lln_theory, col_clt_theory = st.columns(2)

    with col_lln_theory:
        st.markdown("### The Law of Large Numbers (LLN)")
        st.markdown("**Intuitive Definition:** If you repeatedly sample from a stable environment, the average of your observations will eventually lock onto the true mathematical average of that environment. As your sample size grows, the noise of individual random events cancels out.")
        st.markdown("**Mathematical Definition:** Let $X_1, X_2, \dots, X_n$ be a sequence of random variables with a true expected value $\mu = E[X]$. Let $\\bar{X}_n$ be the sample mean:")
        st.latex(r"\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i")
        st.markdown("The Weak Law of Large Numbers states that for any margin of error $\epsilon > 0$, the probability that the sample mean deviates from the true mean approaches zero as $n \\to \infty$:")
        st.latex(r"\lim_{n \to \infty} P(\vert\bar{X}_n - \mu\vert \ge \epsilon) = 0")
        
        st.markdown("""
        **Strict Requirements for the LLN:**
        * **Finite First Moment ($E[|X|] < \infty$):** The core requirement. If the mean is undefined (e.g., Cauchy distribution), the sample average will endlessly jump and never converge.
        * **Identically Distributed & Independent:** Must come from the same probability distribution without influencing each other.
        """)

    with col_clt_theory:
        st.markdown("### The Central Limit Theorem (CLT)")
        st.markdown("While the LLN dictates *where* the sample mean heads, the CLT dictates the *shape* of the errors around that mean.")
        st.markdown("**Mathematical Definition:** Let $X_1, X_2, \dots, X_n$ be a sequence of independent and identically distributed (i.i.d.) random variables with a true expected value $\mu = E[X]$ and a strictly finite variance $\sigma^2 = Var(X) < \infty$.")
        st.markdown("The Average of Sample Variables converges to a Normal distribution centered on the true mean:")
        st.latex(r"\bar{X}_n \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)")
        st.markdown("The Standardized Average ($Z_n$) converges exactly in distribution to the Standard Normal:")
        st.latex(r"Z_n = \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1)")
        
        st.markdown("""
        **Strict Requirements for the CLT:**
        * **Finite Variance ($\sigma^2 < \infty$):** The core prerequisite. The variance dictates the scaling factor ($\sigma / \sqrt{n}$) in the formula.
        * **Finite Mean ($\mu$ exists):** Centering the data is impossible without a defined mean.
        """)

    st.markdown("### Convergence of Specific Sample Statistics")
    st.markdown("""
| Statistic | LLN Convergence | LLN Requirement | CLT Limiting Distribution | CLT Requirement |
| :--- | :--- | :--- | :--- | :--- |
| **Sample average** | Yes | Finite Mean | Normal ($\mathcal{N}$) | Finite Variance |
| **Sample median** | Yes | CDF strictly increasing at median | Normal ($\mathcal{N}$) | Positive density at median |
| **Sample percentiles** | Yes | Glivenko-Cantelli theorem | Normal ($\mathcal{N}$) | Bahadur representation |
| **Sample standard dev.** | Yes | Finite Variance | Normal ($\mathcal{N}$) | Finite 4th moment (Kurtosis) |
| **Sample skewness**| **Conditional**| Finite 3rd moment | **Conditional** ($\mathcal{N}$) | Finite 6th moment (Breaks fast) |
| **Sample kurtosis**| **Conditional**| Finite 4th moment | **Conditional** ($\mathcal{N}$) | Finite 8th moment (Almost never exists) |
| **Sample min & max** | **No** | *Extreme Value Theory (EVT)* | **No** (GEV Distribution) | *Extreme Value Theory (EVT)* |
    """)

# --- Configuration & Data Dictionaries ---
DISTRIBUTIONS = {
    'normal': 'Normal (Thin Tail)',
    'bernoulli-0.5': 'Bernoulli, p=0.5 (Centered) (Discrete, Thin Tail)',
    'geometric-0.5': 'Geometric, p=0.5 (Centered) (Discrete, Thin Tail)',
    'poisson-5': 'Poisson, λ=5 (Centered) (Discrete, Thin Tail)',
    'exponential': 'Exponential (Centered) (Skewed, Thin Tail)',
    'weibull-1.5': 'Weibull, k=1.5 (Centered) (Mild Skew, Thin Tail)',
    'student-t-30': 'Student-t, df=30 (Almost Normal)',
    'student-t-5': 'Student-t, df=5 (Mildly Fat Tail)',
    'student-t-4': 'Student-t, df=4 (Fat Tail)',
    'student-t-3': 'Student-t, df=3 (Fat Tail, Finite Variance)',
    'student-t-2.75': 'Student-t, df=2.75 (Fat Tail, Finite Variance)',
    'student-t-2.5': 'Student-t, df=2.5 (Fat Tail, Finite Variance)',
    'student-t-2.25': 'Student-t, df=2.25 (Fat Tail, Finite Variance)',
    'student-t-2': 'Student-t, df=2 (Infinite Variance)',
    'student-t-1.75': 'Student-t, df=1.75 (Infinite Variance)',
    'student-t-1.5': 'Student-t, df=1.5 (Infinite Variance)',
    'student-t-1.25': 'Student-t, df=1.25 (Infinite Variance)',
    'student-t-1.16': 'Student-t, df=1.16 (Infinite Variance)',
    'pareto-1.75': 'Pareto, α=1.75 (Centered) (Fat Tail, Inf. Var)',
    'pareto-1.5': 'Pareto, α=1.5 (Centered) (Fat Tail, Inf. Var)',
    'pareto-1.25': 'Pareto, α=1.25 (Centered) (Fat Tail, Inf. Var)',
    'pareto-1.16': 'Pareto, α=1.16 (80/20 Principle) (Extreme Fat Tail)',
    'cauchy': 'Cauchy (Unruly, Undefined Mean)'
}

STATISTICS = {
    'mean': 'Sample Mean',
    'variance': 'Sample Variance',
    'skewness': 'Sample Skewness (3rd Moment)',
    'kurtosis': 'Sample Kurtosis (4th Moment)',
    'min': 'Sample Minimum',
    'max': 'Sample Maximum',
    'p1': '1st Percentile',
    'p5': '5th Percentile',
    'p10': '10th Percentile',
    'p25': '25th Percentile',
    'median': 'Median (50th Percentile)',
    'p75': '75th Percentile',
    'p90': '90th Percentile',
    'p95': '95th Percentile',
    'p99': '99th Percentile'
}

# Population Statistics Dictionary (Hardcoded true values)
pop_stats = {
    'normal': {'mean': 0, 'variance': 1, 'skewness': 0, 'kurtosis': 0, 'median': 0, 'p1': -2.326, 'p5': -1.645, 'p10': -1.282, 'p25': -0.674, 'p75': 0.674, 'p90': 1.282, 'p95': 1.645, 'p99': 2.326},
    'bernoulli-0.5': {'mean': 0, 'variance': 0.25, 'skewness': 0, 'kurtosis': -2.0, 'median': 0.5, 'p1': -0.5, 'p5': -0.5, 'p10': -0.5, 'p25': -0.5, 'p75': 0.5, 'p90': 0.5, 'p95': 0.5, 'p99': 0.5},
    'geometric-0.5': {'mean': 0, 'variance': 2, 'skewness': 2.12, 'kurtosis': 6.5, 'median': -1, 'p1': -1, 'p5': -1, 'p10': -1, 'p25': -1, 'p75': 0, 'p90': 2, 'p95': 3, 'p99': 5},
    'poisson-5': {'mean': 0, 'variance': 5, 'skewness': 0.447, 'kurtosis': 0.2, 'median': 0, 'p1': -5, 'p5': -4, 'p10': -3, 'p25': -2, 'p75': 1, 'p90': 3, 'p95': 4, 'p99': 6},
    'exponential': {'mean': 0, 'variance': 1, 'skewness': 2, 'kurtosis': 6, 'median': -np.log(0.5) - 1, 'p1': -np.log(0.99) - 1, 'p5': -np.log(0.95) - 1, 'p10': -np.log(0.90) - 1, 'p25': -np.log(0.75) - 1, 'p75': -np.log(0.25) - 1, 'p90': -np.log(0.10) - 1, 'p95': -np.log(0.05) - 1, 'p99': -np.log(0.01) - 1},
    'weibull-1.5': {'mean': 0, 'variance': 0.3757, 'skewness': 1.072, 'kurtosis': 1.39, 'median': -0.1195, 'p1': -0.856, 'p5': -0.765, 'p10': -0.680, 'p25': -0.466, 'p75': 0.339, 'p90': 0.840, 'p95': 1.176, 'p99': 1.863},
    'student-t-30': {'mean': 0, 'variance': 30/28, 'skewness': 0, 'kurtosis': 6/26, 'median': 0, 'p1': -2.457, 'p5': -1.697, 'p10': -1.310, 'p25': -0.683, 'p75': 0.683, 'p90': 1.310, 'p95': 1.697, 'p99': 2.457},
    'student-t-5': {'mean': 0, 'variance': 5/3, 'skewness': 0, 'kurtosis': 6, 'median': 0, 'p1': -3.365, 'p5': -2.015, 'p10': -1.476, 'p25': -0.727, 'p75': 0.727, 'p90': 1.476, 'p95': 2.015, 'p99': 3.365},
    'student-t-4': {'mean': 0, 'variance': 2, 'skewness': 0, 'kurtosis': np.inf, 'median': 0, 'p1': -3.747, 'p5': -2.132, 'p10': -1.533, 'p25': -0.741, 'p75': 0.741, 'p90': 1.533, 'p95': 2.132, 'p99': 3.747},
    'student-t-3': {'mean': 0, 'variance': 3, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -4.541, 'p5': -2.353, 'p10': -1.638, 'p25': -0.765, 'p75': 0.765, 'p90': 1.638, 'p95': 2.353, 'p99': 4.541},
    'student-t-2.75': {'mean': 0, 'variance': 11/3, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -4.939, 'p5': -2.463, 'p10': -1.693, 'p25': -0.777, 'p75': 0.777, 'p90': 1.693, 'p95': 2.463, 'p99': 4.939},
    'student-t-2.5': {'mean': 0, 'variance': 5.0, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -5.452, 'p5': -2.607, 'p10': -1.758, 'p25': -0.788, 'p75': 0.788, 'p90': 1.758, 'p95': 2.607, 'p99': 5.452},
    'student-t-2.25': {'mean': 0, 'variance': 9.0, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -6.115, 'p5': -2.784, 'p10': -1.832, 'p25': -0.801, 'p75': 0.801, 'p90': 1.832, 'p95': 2.784, 'p99': 6.115},
    'student-t-2': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -6.965, 'p5': -2.920, 'p10': -1.886, 'p25': -0.816, 'p75': 0.816, 'p90': 1.886, 'p95': 2.920, 'p99': 6.965},
    'student-t-1.75': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -8.571, 'p5': -3.220, 'p10': -1.996, 'p25': -0.835, 'p75': 0.835, 'p90': 1.996, 'p95': 3.220, 'p99': 8.571},
    'student-t-1.5': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -11.196, 'p5': -3.655, 'p10': -2.146, 'p25': -0.861, 'p75': 0.861, 'p90': 2.146, 'p95': 3.655, 'p99': 11.196},
    'student-t-1.25': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -16.488, 'p5': -4.364, 'p10': -2.366, 'p25': -0.896, 'p75': 0.896, 'p90': 2.366, 'p95': 4.364, 'p99': 16.488},
    'student-t-1.16': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': -19.988, 'p5': -4.755, 'p10': -2.476, 'p25': -0.912, 'p75': 0.912, 'p90': 2.476, 'p95': 4.755, 'p99': 19.988},
    'pareto-1.75': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': np.power(0.5, -1/1.75) - 7/3, 'p1': np.power(0.99, -1/1.75) - 7/3, 'p5': np.power(0.95, -1/1.75) - 7/3, 'p10': np.power(0.90, -1/1.75) - 7/3, 'p25': np.power(0.75, -1/1.75) - 7/3, 'p75': np.power(0.25, -1/1.75) - 7/3, 'p90': np.power(0.10, -1/1.75) - 7/3, 'p95': np.power(0.05, -1/1.75) - 7/3, 'p99': np.power(0.01, -1/1.75) - 7/3},
    'pareto-1.5': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': np.power(0.5, -2/3) - 3, 'p1': np.power(0.99, -2/3) - 3, 'p5': np.power(0.95, -2/3) - 3, 'p10': np.power(0.90, -2/3) - 3, 'p25': np.power(0.75, -2/3) - 3, 'p75': np.power(0.25, -2/3) - 3, 'p90': np.power(0.10, -2/3) - 3, 'p95': np.power(0.05, -2/3) - 3, 'p99': np.power(0.01, -2/3) - 3},
    'pareto-1.25': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': np.power(0.5, -1/1.25) - 5, 'p1': np.power(0.99, -1/1.25) - 5, 'p5': np.power(0.95, -1/1.25) - 5, 'p10': np.power(0.90, -1/1.25) - 5, 'p25': np.power(0.75, -1/1.25) - 5, 'p75': np.power(0.25, -1/1.25) - 5, 'p90': np.power(0.10, -1/1.25) - 5, 'p95': np.power(0.05, -1/1.25) - 5, 'p99': np.power(0.01, -1/1.25) - 5},
    'pareto-1.16': {'mean': 0, 'variance': np.inf, 'skewness': np.nan, 'kurtosis': np.nan, 'median': np.power(0.5, -1/1.16) - 7.25, 'p1': np.power(0.99, -1/1.16) - 7.25, 'p5': np.power(0.95, -1/1.16) - 7.25, 'p10': np.power(0.90, -1/1.16) - 7.25, 'p25': np.power(0.75, -1/1.16) - 7.25, 'p75': np.power(0.25, -1/1.16) - 7.25, 'p90': np.power(0.10, -1/1.16) - 7.25, 'p95': np.power(0.05, -1/1.16) - 7.25, 'p99': np.power(0.01, -1/1.16) - 7.25},
    'cauchy': {'mean': np.nan, 'variance': np.nan, 'skewness': np.nan, 'kurtosis': np.nan, 'median': 0, 'p1': np.tan(np.pi * (0.01 - 0.5)), 'p5': np.tan(np.pi * (0.05 - 0.5)), 'p10': np.tan(np.pi * (0.10 - 0.5)), 'p25': -1, 'p75': 1, 'p90': np.tan(np.pi * (0.90 - 0.5)), 'p95': np.tan(np.pi * (0.95 - 0.5)), 'p99': np.tan(np.pi * (0.99 - 0.5))}
}

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Simulation Controls")
    
    selected_dist_key = st.selectbox(
        "Distribution (Machine)",
        options=list(DISTRIBUTIONS.keys()),
        format_func=lambda x: DISTRIBUTIONS[x]
    )
    
    selected_stat_key = st.selectbox(
        "Statistic to Track",
        options=list(STATISTICS.keys()),
        format_func=lambda x: STATISTICS[x]
    )
    
    n_clt = st.number_input("Sample Size (n) per Trial (CLT)", min_value=5, max_value=100000, value=1000, step=5)
    
    n_lln = st.number_input("Total Samples (LLN limit)", min_value=10, max_value=1000000, value=1000, step=10) 
    
    st.subheader("Fixed Chart Range Bounds")
    col1, col2 = st.columns(2)
    with col1:
        y_min = st.number_input("Min", value=-1.0, step=0.5)
    with col2:
        y_max = st.number_input("Max", value=1.0, step=0.5)
        
    run_simulation = st.button("Run Simulation", type="primary", use_container_width=True)

# --- Generator Functions ---
def generate_samples(dist_key, size):
    if dist_key == 'normal':
        return norm.rvs(size=size)
    elif dist_key == 'bernoulli-0.5':
        return bernoulli.rvs(p=0.5, size=size) - 0.5
    elif dist_key == 'geometric-0.5':
        return geom.rvs(p=0.5, size=size) - 2 
    elif dist_key == 'poisson-5':
        return poisson.rvs(mu=5, size=size) - 5
    elif dist_key == 'exponential':
        return expon.rvs(size=size) - 1.0
    elif dist_key == 'weibull-1.5':
        mean_weibull = 0.9027452929509337
        return weibull_min.rvs(c=1.5, scale=1.0, size=size) - mean_weibull
    elif dist_key.startswith('student-t-'):
        df = float(dist_key.split('-')[2])
        # Z / sqrt(V/df) where V ~ Gamma(df/2, 2)
        return t.rvs(df=df, size=size)
    elif dist_key.startswith('pareto-'):
        alpha = float(dist_key.split('-')[1])
        theoretical_mean = (alpha / (alpha - 1)) if alpha > 1 else 0
        return pareto.rvs(b=alpha, size=size) - theoretical_mean
    elif dist_key == 'cauchy':
        return cauchy.rvs(size=size)
    return np.zeros(size)

def calculate_statistic(data, stat_key):
    if stat_key == 'mean':
        return np.mean(data)
    elif stat_key == 'variance':
        return np.var(data, ddof=1) if len(data) > 1 else 0
    elif stat_key == 'skewness':
        return skew(data, bias=False) if len(data) > 2 else 0
    elif stat_key == 'kurtosis':
        return kurtosis(data, bias=False) if len(data) > 3 else 0
    elif stat_key == 'min':
        return np.min(data)
    elif stat_key == 'max':
        return np.max(data)
    elif stat_key == 'median':
        return np.percentile(data, 50)
    elif stat_key.startswith('p'):
        perc = float(stat_key[1:])
        return np.percentile(data, perc)
    return 0

# Initialize sample rate for the session state if it doesn't exist
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = 1

# --- Main Logic & Simulation ---
if run_simulation or 'lln_data' not in st.session_state:
    
    # 1. Run LLN Simulation & MS Plot Data
    lln_samples = generate_samples(selected_dist_key, n_lln)
    
    lln_x = []
    lln_y = []
    
    sample_rate = 1
    if n_lln > 10000:
        sample_rate = 100
    elif n_lln > 1000:
        sample_rate = 10
        
    st.session_state.sample_rate = sample_rate
        
    for i in range(1, n_lln + 1):
        if i < 100 or i % sample_rate == 0 or i == n_lln:
            current_slice = lln_samples[:i]
            stat_val = calculate_statistic(current_slice, selected_stat_key)
            lln_x.append(i)
            lln_y.append(stat_val)
            
    st.session_state.lln_data = pd.DataFrame({'n': lln_x, 'value': lln_y})
    
    # Pre-calculate data for Maximum-to-Sum plot
    abs_samples = np.abs(lln_samples)
    st.session_state.ms_data = pd.DataFrame({'n': np.arange(1, n_lln + 1)})
    
    for p in [1, 2, 3, 4]:
        # Force float64 to avoid UFuncTypeError during divide on discrete variables
        pow_samples = np.power(abs_samples, p).astype(np.float64) 
        running_max = np.maximum.accumulate(pow_samples)
        running_sum = np.cumsum(pow_samples)
        out_array = np.zeros_like(running_max, dtype=np.float64)
        ratio = np.divide(running_max, running_sum, out=out_array, where=running_sum!=0)
        st.session_state.ms_data[f'p={p}'] = ratio
        
    # Pre-calculate data for Mean Excess Plot & Zipf Plot
    Z = np.abs(lln_samples)
    Z_sorted = np.sort(Z)
    
    # MEP Data
    k_min = np.percentile(Z_sorted, 50)
    k_max = np.percentile(Z_sorted, 98)
    
    K_vals = np.linspace(k_min, k_max, 100)
    e_K = []
    for k in K_vals:
        excesses = Z_sorted[Z_sorted > k] - k
        e_K.append(np.mean(excesses) if len(excesses) > 0 else np.nan)
        
    st.session_state.mep_data = pd.DataFrame({'K': K_vals, 'e(K)': e_K})
    
    # Zipf Plot Data (Log-Log Survival)
    Z_nonzero = Z_sorted[Z_sorted > 0]
    n_nonzero = len(Z_nonzero)
    if n_nonzero > 0:
        zipf_K = Z_nonzero
        # Survival probability: rank / N
        zipf_P = np.arange(n_nonzero, 0, -1) / n_nonzero
        
        # Subsample for rendering performance
        if n_nonzero > 1000:
            indices = np.linspace(0, n_nonzero - 1, 1000).astype(int)
            zipf_K = zipf_K[indices]
            zipf_P = zipf_P[indices]
            
        st.session_state.zipf_data = pd.DataFrame({'K': zipf_K, 'P(X>K)': zipf_P})
    else:
        st.session_state.zipf_data = pd.DataFrame({'K': [], 'P(X>K)': []})
    
    # 2. Run CLT Simulation
    num_trials = 1000
    clt_matrix = generate_samples(selected_dist_key, num_trials * n_clt).reshape((num_trials, n_clt))
    
    if selected_stat_key == 'mean':
        clt_y = np.mean(clt_matrix, axis=1)
    elif selected_stat_key == 'variance':
        clt_y = np.var(clt_matrix, axis=1, ddof=1)
    elif selected_stat_key == 'skewness':
        clt_y = skew(clt_matrix, axis=1, bias=False)
    elif selected_stat_key == 'kurtosis':
        clt_y = kurtosis(clt_matrix, axis=1, bias=False)
    elif selected_stat_key == 'min':
        clt_y = np.min(clt_matrix, axis=1)
    elif selected_stat_key == 'max':
        clt_y = np.max(clt_matrix, axis=1)
    elif selected_stat_key == 'median':
        clt_y = np.percentile(clt_matrix, 50, axis=1)
    elif selected_stat_key.startswith('p'):
        perc = float(selected_stat_key[1:])
        clt_y = np.percentile(clt_matrix, perc, axis=1)
    else:
        clt_y = np.zeros(num_trials)
        
    st.session_state.clt_data = pd.DataFrame({'value': clt_y})

# --- Rendering Charts ---
st.markdown("<h2 class='section-header' style='margin-top: 1rem;'>Simulation Engine</h2>", unsafe_allow_html=True)

st.markdown("<h3>Law of Large Numbers (LLN)</h3>", unsafe_allow_html=True)

pop_val = pop_stats[selected_dist_key].get(selected_stat_key, np.nan)
pop_label = 'Undefined / Does Not Exist'

if pd.notna(pop_val):
    if np.isinf(pop_val):
        pop_label = 'Infinity'
    else:
        pop_label = str(int(pop_val)) if float(pop_val).is_integer() else f"{pop_val:.3f}"

if selected_stat_key == 'max':
    pop_label = 'Grows infinitely (EVT governed)'
if selected_stat_key == 'min':
    pop_label = 'Decreases infinitely (EVT governed)'

st.markdown(f"<p class='chart-desc'>Tracks statistical convergence by plotting the Sample Size <i>n</i> (X-axis) against the running, cumulative <b>{STATISTICS[selected_stat_key]}</b> (Y-axis). True population value: <b>{pop_label}</b>.</p>", unsafe_allow_html=True)

fig_lln = px.line(st.session_state.lln_data, x='n', y='value')
fig_lln.update_traces(line_color='#2563eb', line_width=1.5)

if pd.notna(pop_val) and not np.isinf(pop_val) and selected_stat_key not in ['max', 'min']:
    fig_lln.add_hline(y=pop_val, line_dash="dash", line_color="#ef4444", line_width=2)

fig_lln.update_layout(
    xaxis_title="Sample Size (n) \u2192",
    yaxis_title=f"Cumulative Sample {STATISTICS[selected_stat_key]} \u2191",
    yaxis=dict(range=[y_min, y_max], constrain='domain'),
    margin=dict(l=40, r=20, t=20, b=40),
    height=400,
    plot_bgcolor='white',
    paper_bgcolor='white'
)
fig_lln.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
fig_lln.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')

st.plotly_chart(fig_lln, use_container_width=True)

st.markdown("<h3>Central Limit Theorem (CLT)</h3>", unsafe_allow_html=True)
st.markdown(f"<p class='chart-desc'>Tests the assumption of finite variance by taking 1,000 independent trials of size <i>n={n_clt}</i>, and plotting the resulting <b>{STATISTICS[selected_stat_key]}</b> (X-axis) against its Frequency (Y-axis).</p>", unsafe_allow_html=True)

valid_clt_data = st.session_state.clt_data[
    (st.session_state.clt_data['value'] >= y_min) & 
    (st.session_state.clt_data['value'] <= y_max)
]

num_bins = 50
bin_step = (y_max - y_min) / num_bins
bins = np.arange(y_min, y_max + bin_step, bin_step)

fig_clt = go.Figure()
fig_clt.add_trace(go.Histogram(
    x=valid_clt_data['value'],
    xbins=dict(start=y_min, end=y_max, size=bin_step),
    marker_color='#38bdf8'
))

fig_clt.update_layout(
    xaxis_title=f"Sample {STATISTICS[selected_stat_key]} \u2192",
    yaxis_title="Frequency \u2191",
    xaxis=dict(range=[y_min, y_max]),
    bargap=0.05,
    margin=dict(l=40, r=20, t=20, b=40),
    height=400,
    plot_bgcolor='white',
    paper_bgcolor='white'
)
fig_clt.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
fig_clt.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')

st.plotly_chart(fig_clt, use_container_width=True)

# --- EMPIRICAL FAT TAIL DIAGNOSTICS SECTION ---
st.markdown("<h2 class='section-header'>Empirical Fat Tail Diagnostics</h2>", unsafe_allow_html=True)
st.markdown("<p class='chart-desc'>These charts analyze the dataset generated in the LLN simulation above to diagnose the severity of the tails. Standard models assume moments exist and extreme events dampen out. In Extremistan, these assumptions break visibly.</p>", unsafe_allow_html=True)

col_left, col_mid, col_right = st.columns(3)

with col_left:
    st.markdown("### Zipf Plot")
    st.caption("""Diagnoses the empirical rate of tail decay by plotting the Log of a threshold $K$ (X-axis) against the Log of the probability of exceeding that threshold, $P(X > K)$ (Y-axis).
*   **Thin Tail:** The line curves and plunges almost vertically downward, indicating that extreme outliers face a strict probabilistic boundary.
*   **Fat Tail:** The line forms a shallow, negatively sloping diagonal, indicating extreme events decay slowly and remain highly probable.""")
    
    fig_zipf = px.line(st.session_state.zipf_data, x='K', y='P(X>K)')
    fig_zipf.update_traces(line_color='#db2777', line_width=2)
    
    fig_zipf.update_layout(
        xaxis_type="log",
        yaxis_type="log",
        xaxis_title="Threshold (K) [Log] \u2192",
        yaxis_title="P(X > K) [Log] \u2191",
        margin=dict(l=40, r=20, t=20, b=40),
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig_zipf.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False)
    fig_zipf.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=False)
    
    st.plotly_chart(fig_zipf, use_container_width=True)

with col_mid:
    st.markdown("### Maximum-to-Sum Plot")
    st.caption("""Visually tests if a moment is finite by plotting sample size $n$ (X-axis) against the ratio of the single maximum observation to the sum of all observations, $\\frac{\\max(|X|^p)}{\\sum |X|^p}$ (Y-axis), for moments $p=1, 2, 3, 4$.
*   **Thin Tail:** All moment lines smoothly drop toward 0 as sample size grows, indicating the moments are finite and stable.
*   **Fat Tail:** The lines for higher moments (like $p=2, 3, 4$) jump erratically or hover above 0 indefinitely, indicating a single outlier dominates the dataset and the moment is mathematically infinite.""")
    
    fig_ms = go.Figure()
    colors = {1: '#3b82f6', 2: '#10b981', 3: '#f59e0b', 4: '#ef4444'}
    for p in [1, 2, 3, 4]:
        current_sr = st.session_state.get('sample_rate', 1)
        plot_df = st.session_state.ms_data[st.session_state.ms_data['n'] % current_sr == 0]
        fig_ms.add_trace(go.Scatter(
            x=plot_df['n'], 
            y=plot_df[f'p={p}'], 
            mode='lines', 
            name=f'p={p} (Moment {p})',
            line=dict(color=colors[p], width=1.5)
        ))
        
    fig_ms.update_layout(
        xaxis_title="Sample Size (n) \u2192",
        yaxis_title="Max / Sum Ratio \u2191",
        yaxis=dict(range=[-0.05, 1.05]),
        margin=dict(l=40, r=20, t=20, b=40),
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor="rgba(255,255,255,0.8)")
    )
    fig_ms.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
    fig_ms.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
    
    st.plotly_chart(fig_ms, use_container_width=True)

with col_right:
    st.markdown("### Mean Excess Plot")
    st.caption("""Plots how severe deviations become *after* an initial threshold $K$ (X-axis) is breached, measured against the expected excess beyond $K$, $e(K) = E[|X| - K \mid |X| > K]$ (Y-axis).
*   **Thin Tail:** The line slopes downward, indicating that once a threshold is breached, subsequent deviations are pulled aggressively back toward normal levels.
*   **Fat Tail:** The line slopes upward, indicating an acceleration effect: once an extreme threshold is breached, the expected severity of the event continues to grow.""")
    
    fig_mep = px.line(st.session_state.mep_data, x='K', y='e(K)')
    fig_mep.update_traces(line_color='#8b5cf6', line_width=2)
    
    fig_mep.update_layout(
        xaxis_title="Threshold (K) \u2192",
        yaxis_title="Expected Excess e(K) \u2191",
        margin=dict(l=40, r=20, t=20, b=40),
        height=350,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    fig_mep.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
    fig_mep.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#e2e8f0', zeroline=True, zerolinecolor='#cbd5e1')
    
    st.plotly_chart(fig_mep, use_container_width=True)