# Quantitative Option Valuation & Risk Profiling Tool

A professional-grade, Streamlit-based web application for advanced quantitative options pricing and structural risk analysis. 

This tool bridges the gap between traditional theoretical finance and empirical, path-dependent market reality. It evaluates multi-leg option structures by progressing from rigid Lognormal assumptions (Standard Black-Scholes) to semi-parametric Filtered Historical Simulations using Extreme Value Theory (EVT) and GARCH volatility clustering. 

It natively supports Equities, FX (via Garman-Kohlhagen adaptations setting $q$ to the foreign risk-free rate), and VIX derivatives (via Black's 1976 adaptations setting $q = r$).

---

## 🧠 The Pricing Models & Mathematical Formulations

The core of this application relies on an evolutionary hierarchy of four pricing models. It progressively strips away theoretical assumptions to reveal physical market realities.

### 1. Standard Black-Scholes (Analytical Constant Volatility)
**The Intuition:** Values the option analytically assuming volatility is perfectly constant and returns follow a smooth Lognormal distribution (no Black Swans).
**Implementation:** Uses the closed-form analytical Black-Scholes formula, substituting the risk-free rate ($r$) with the subjective physical drift ($\mu$) inside the probabilities for physical state mapping.

**The Math (Distributions & Drift):**
* **Adjusted Drift:** $\mu_{adj} = \mu - q - \frac{1}{2}\sigma^2$
* **1-Period Return:** $\ln\left(\frac{S_{t+\Delta t}}{S_t}\right) \sim N(\mu_{adj} \Delta t, \sigma^2 \Delta t)$
* **Terminal Price Distribution:** $S_T \sim \text{Lognormal}(\ln S_0 + \mu_{adj} T, \sigma^2 T)$
* **Terminal Price ($Z$):** $S_T = S_0 \exp(\mu_{adj} T + \sigma \sqrt{T} Z)$ where $Z \sim N(0,1)$

### 2. Black-Scholes Mixture Model (Bootstrapped Empirical Volatility)
**The Intuition:** Recognizes volatility clustering. Instead of assuming one constant volatility, it repeatedly samples from historically realized $N$-day volatility regimes.
**Implementation:** Monte Carlo simulation. The engine draws a historical volatility $\sigma_i$ at random, evaluates the analytical formula using that specific volatility, repeats 10,000 times, and averages the outcomes to create a fat-tailed "Mixture Distribution."

**The Math (Distributions & Drift):**
* **Adjusted Drift for Sample $i$:** $\mu_{adj, i} = \mu - q - \frac{1}{2}\sigma_i^2$
* **Terminal Price ($Z$):** $S_T = S_0 \exp(\mu_{adj, i} T + \sigma_i \sqrt{T} Z)$ where $Z \sim N(0,1)$

**Why This Model Prices Higher (Jensen's Inequality):**
Option prices are strictly convex to volatility (positive Volga/Vomma). A massive volatility spike adds more dollar value to the option than a massive volatility drop takes away. According to **Jensen's Inequality** ($E[f(X)] \ge f(E[X])$), the average of the option prices across varied volatilities will always be greater than the option price calculated using a single average volatility.

### 3. Return-Based Extreme Value Theory (EVT) Model
**The Intuition:** Abandons Normal distribution assumptions entirely. Simulates daily steps using exact historical returns, augmented with Extreme Value Theory (EVT) to accurately model physical "Black Swan" jumps.
**Implementation:** Simulates the path day-by-day drawing a uniform random variable $U \sim U(0,1)$. If $U < 0.05$, it draws a massive crash from the Left GPD tail. If $U > 0.95$, it draws a squeeze from the Right GPD tail. Otherwise, it samples a normal historical day.

**The Math (Distributions & Drift):**
* **Adjusted Drift:** $\mu_{adj} = \mu - q - \frac{1}{2}\sigma^2$
* **1-Period Return:** $R_t = \ln\left(\frac{S_t}{S_{t-1}}\right) \sim F_{EVT}$ (Empirical Center + GPD Tails)
* **Terminal Price:** $S_T = S_0 \exp\left( \sum_{t=1}^{N} R_t + \mu_{adj} T \right)$

**Understanding the Shape Parameters ($\xi$):**
* **Left Tail Shape Parameter ($\xi$):** Quantifies the "fatness" of market crashes. 
  * $\xi \le 0$: Thin tails (Normal risk).
  * $0 < \xi < 0.5$: Heavy tails. Crashes are far more likely than a Normal distribution assumes.
  * $\xi \ge 0.5$: Infinite Variance. Historical worst-case scenarios do not limit future worst-case scenarios. A higher number implies severe put option premiums.
* **Right Tail Shape Parameter ($\xi$):** Quantifies the "fatness" of market squeezes and euphoric rallies. Usually lower than the Left Tail index, but spikes indicate severe upside tail risk (e.g., a short squeeze regime).

### 4. GARCH-EVT Filtered Historical Simulation (FHS)
**The Intuition:** The gold standard for path dependency. Recognizes that an extreme shock today massively spikes the baseline volatility tomorrow (GARCH), creating sustained market panic.
**Implementation:** Starts with today's volatility. Draws an EVT shock ($Z_t$), calculates the physical return ($R_t$), and critically, feeds that shock into the GARCH equation to update tomorrow's volatility ($\sigma_t$). A massive shock early forces violent thrashing for the remaining path.

**The Math (Distributions & Drift):**
* **1-Period Return:** $R_t = \mu_t + \sigma_{t-1} Z_t$
* **The Shock (Innovation):** $Z_t \sim F_{EVT}$ (EVT fitted specifically to standardized shocks)
* **Dynamic Variance:** $\sigma_t^2 = \omega + \alpha (R_{t-1} - \mu_{t-1})^2 + \beta \sigma_{t-1}^2$
* **Dynamic Drift:** $\mu_t = (\mu - q - \frac{1}{2}\sigma_{t-1}^2)\Delta t$
* **Terminal Price:** $S_T = S_0 \exp\left( \sum_{t=1}^{N} R_t \right)$

---

## 📊 Option Greeks Summary Guide

The tool provides an integrated 2D and 3D Risk Profiler that computes sensitivity metrics across the spot price and volatility surfaces.

### First-Order Greeks (Direct Sensitivities)
* **Delta ($\Delta$):** $\frac{\partial V}{\partial S}$ — Directional risk. Expected change in option value per $\$1.00$ change in the underlying asset price. Also acts as a proxy for In-The-Money probability.
* **Vega ($\nu$):** $\frac{\partial V}{\partial \sigma}$ — Volatility risk. Expected change in option value per $1\%$ absolute change in implied volatility. Long options are always long Vega.
* **Theta ($\Theta$):** $\frac{\partial V}{\partial t}$ — Time decay. Expected daily dollar loss in option value strictly due to the passage of time. Accelerates toward expiration.
* **Rho ($\rho$):** $\frac{\partial V}{\partial r}$ — Interest rate risk. Sensitivity to a $1\%$ change in the risk-free interest rate.

### Second-Order Greeks (Convexity & Acceleration)
* **Gamma ($\Gamma$):** $\frac{\partial^2 V}{\partial S^2}$ or $\frac{\partial \Delta}{\partial S}$ — Directional acceleration. Rate of change in Delta per $\$1.00$ change in stock price. Highly concentrated At-The-Money.
* **Vomma / Volga:** $\frac{\partial^2 V}{\partial \sigma^2}$ or $\frac{\partial \nu}{\partial \sigma}$ — Volatility convexity. Rate of change in Vega per $1\%$ change in implied volatility. Captures the upward curve of option premium under extreme fear.
* **Vanna:** $\frac{\partial^2 V}{\partial S \partial \sigma}$ — Cross-derivative. Rate of change in Delta per $1\%$ change in volatility (or Vega per $\$1.00$ spot move). Shows how volatility spikes drag Out-Of-The-Money options closer to the money.
* **Charm:** $\frac{\partial^2 V}{\partial S \partial t}$ — Delta decay. The absolute daily change in Delta due to the passage of time. Magnetizes OTM Delta to 0 and ITM Delta to 1.

---

## 💻 Installation & Cloud Deployment

### Local Usage
**1. Clone the repository:**
```bash
git clone [https://github.com/yourusername/option-valuation-tool.git](https://github.com/yourusername/option-valuation-tool.git)
cd option-valuation-tool