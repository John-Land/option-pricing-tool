# Quantitative Option Valuation & Risk Profiling Tool

A professional-grade, Streamlit-based web application for advanced quantitative options pricing and structural risk analysis. 

This tool bridges the gap between traditional theoretical finance and empirical, path-dependent market reality. It allows traders and quantitative researchers to evaluate multi-leg option structures by progressing from rigid Lognormal assumptions (Standard Black-Scholes) all the way to semi-parametric Filtered Historical Simulations using Extreme Value Theory (EVT) and GARCH volatility clustering.

---

## 🚀 Key Features

*   **Multi-Leg Strategy Engine:** Construct complex structural payoffs (Straddles, Condors, Butterflies, Custom Synthetics) with automatic directional scaling.
*   **Four Distinct Pricing Engines:** Evaluate premiums across four progressive mathematical models to isolate structural mispricing and tail risk.
*   **Dynamic Risk Profiler:** Real-time computation of First-Order ($\Delta, \nu, \Theta, \rho$) and Second-Order ($\Gamma$, Vomma, Vanna) Greeks.
*   **2D & 3D Surface Visualizations:** Interactive Plotly charts mapping time decay (Charm) and empirical volatility shocks across 3D risk topographies.
*   **Automated Data Ingestion:** Natively fetches maximum available historical daily data via `yfinance` to automatically calibrate empirical volatility arrays and GARCH parameters.

---

## 🧠 The Pricing Models

The core of this application relies on an evolutionary hierarchy of pricing models, progressively stripping away theoretical assumptions to reveal physical market realities.

### 1. Standard Black-Scholes (Analytical Constant Volatility)
*   **The Premise:** Assumes asset returns follow a smooth, continuous Lognormal distribution, and that volatility ($\sigma$) remains perfectly constant until expiration.
*   **When it works:** Highly accurate for short-term, At-The-Money (ATM) options and establishing instantaneous Delta-hedging baselines.
*   **The Blindspot:** Mathematically assumes extreme crashes are virtually impossible. It severely underprices deep Out-Of-The-Money (OTM) options, ignores volatility skew, and fails on mean-reverting assets.

### 2. Black-Scholes Mixture Model (Bootstrapped Empirical Volatility)
*   **The Premise:** Recognizes that market volatility shifts across different historical regimes. It runs a Monte Carlo simulation, drawing a random historical $N$-day rolling volatility for each path, and applies it to the analytical formula.
*   **When it works:** Excellent for capturing Volatility Convexity (Vomma/Volga). Due to Jensen’s Inequality, averaging option prices across varied extreme volatilities mathematically inflates the fair value, natively pricing in volatility uncertainty.
*   **The Blindspot:** It still assumes a perfectly smooth Lognormal distribution *within* each individual simulated path and lacks day-to-day sequential path dependency.

### 3. Return-Based Extreme Value Theory (EVT) Model
*   **The Premise:** Abandons the Normal distribution entirely. It steps forward day-by-day, drawing daily returns from historical empirical data. It isolates the extreme tails (e.g., worst 5% of crashes) and fits them to a **Generalized Pareto Distribution (GPD)** to accurately simulate physical Black Swan jumps.
*   **When it works:** The ultimate tool for pricing structural tail risk and asymmetrical assets (e.g., markets that crash faster than they rally). The Left and Right GPD Shape Parameters ($\xi$) reveal the true physical thickness of the tails.
*   **The Blindspot:** Assumes daily returns are Independent and Identically Distributed (i.i.d.). It ignores volatility clustering (the reality that extreme crashes breed sustained high-volatility turbulence).

### 4. GARCH-EVT Filtered Historical Simulation (FHS)
*   **The Premise:** The gold standard for path-dependent risk. It utilizes a **GARCH(1,1)** process to filter historical returns into standardized shocks, and fits EVT strictly to those pure shocks. During the forward Monte Carlo simulation, if a massive EVT shock is drawn, the GARCH equation dynamically spikes the baseline volatility for the subsequent simulated days.
*   **When it works:** Organically replicates prolonged market panics, clustered tail risks, and the persistent expansion of Vega. Highly effective for pricing FX Options and VIX Options where regime persistence is paramount.
*   **The Blindspot:** Highly sensitive to structural regime shifts. If the asset fundamentally changes its behavior, historical GARCH memory may misprice future mean-reversion speed.

---

## 💻 Installation & Local Usage

**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/option-valuation-tool.git](https://github.com/yourusername/option-valuation-tool.git)
cd option-valuation-tool