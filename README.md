# Quantitative Option Valuation & Fat-Tail Risk Analysis Tool

A high-performance quantitative finance and options engineering application built with Streamlit, Plotly, SciPy, and NumPy. 

This platform replaces the Gaussian and constant-volatility assumptions of classical financial engineering with non-linear volatility dynamics and Extreme Value Theory (EVT). It evaluates single-leg options and complex multi-leg combinations across physical real-world probability distributions (P) rather than artificial risk-neutral surfaces (Q), capturing volatility clustering, asymmetric tail decay, parameter estimation risk, and Jensen's inequality convexity.

---

## 📐 Theoretical Framework: Physical Valuation vs. Risk-Neutral Pricing

Standard option pricing models operate under the risk-neutral measure, discounting expected terminal payoffs at the risk-free rate under the assumption that all underlying assets drift at the risk free rate minus dividend yield. While mathematically convenient for dynamic delta-hedging in complete markets, risk-neutral pricing fails to reflect:

1. **Physical Expectation:** Real-world assets compound with an equity risk premium (ERP) and idiosyncratic alpha: 
   
```math
\mu = r + \beta(\text{ERP}) + \alpha_{\text{drift}}/T
```

2. **Paretian Tail Power Laws:** Equity crash distributions exhibit heavy tails characterized by Generalized Pareto Distributions where extreme outcomes decay as power laws rather than exponentially as in Gaussian models.

3. **Volatility Convexity (Jensen's Inequality):** Because option contracts possess positive Vega and Vomma, integrating over an uncertain or fluctuating volatility distribution yields an expected value strictly greater than evaluating the contract at the mean volatility:
   
```math
\mathbb{E}[V(\sigma)] \ge V(\mathbb{E}[\sigma])
```

---

## 🧠 Comprehensive Model Hierarchy & Mathematical Formulations

The tool implements an evolutionary hierarchy of seven valuation engines. Each engine progressively strips away artificial constraints to model real-world asset price dynamics.

```text
[1. Standard Black-Scholes] 
       │ (Relax constant volatility)
       ▼
[2. Empirical Bootstrap Mixture] 
       │ (Overcome historical max volatility truncation)
       ▼
[3. Hybrid Mixture (EVT Volatility Tail)] 
       │ (Transition from macro-vol mixture to daily return simulation)
       ▼
[4. Return-Based EVT (Empirical Body + GPD Tails)] 
       │ (Filter dependent extreme clusters via Runs Method)
       ▼
[5. Declustered EVT (Runs Method Block Maxima)] 
       │ (Introduce autoregressive conditional heteroskedasticity)
       ▼
[6. GARCH(1,1)-EVT Filtered Historical Simulation (L2 Norm)] 
       │ (Overcome infinite fourth-moment instability of squared returns)
       ▼
[7. MAD-Based Filtered Historical Simulation (L1 Norm + EVT)]
```

---

### Model 1: Standard Black-Scholes (Analytical Constant Volatility)

#### 1. Conceptual Intuition
Values the option structure analytically under the classical Black-Scholes-Merton assumption that the underlying asset follows continuous Geometric Brownian Motion (GBM) with constant volatility and constant drift. It ignores fat tails, skew, and market shocks.

#### 2. Probabilistic Notation & Distributional Assumptions
Under the physical probability measure, the continuous price process satisfies the Stochastic Differential Equation (SDE):

```math
dS_t = (\mu - q) S_t \, dt + \sigma S_t \, dW_t
```

By Itô's Lemma, the log return over horizon T follows a Gaussian distribution:

```math
\ln\left(\frac{S_T}{S_0}\right) \sim \mathcal{N}\left(\left(\mu - q - \frac{1}{2}\sigma^2\right)T, \, \sigma^2 T\right)
```

The terminal price is distributed lognormally:

```math
S_T \sim \text{Lognormal}\left(\ln S_0 + \mu_{adj} T, \, \sigma^2 T\right)
```

#### 3. Calibration & Parameter Estimation
* **Volatility:** Sample standard deviation of historical daily log returns scaled to an annual horizon:
  
```math
\sigma = \sqrt{\frac{252}{N-1} \sum_{k=1}^N (R_k - \bar{R})^2}
```

* **Physical Drift:** User-calibrated Capital Asset Pricing Model (CAPM) structure:
  
```math
\mu = r + \beta \cdot \text{ERP} + \frac{\alpha_{\text{drift}}}{T}
```

#### 4. Valuation & Simulation Process
The terminal intrinsic value for an arbitrary set of call/put legs is evaluated analytically:

```math
V_0 = e^{-rT} \sum_{j=1}^M w_j \left[ \phi_j S_0 e^{(\mu - q)T} \mathcal{N}(\phi_j d_{1,j}) - \phi_j K_j \mathcal{N}(\phi_j d_{2,j}) \right]
```

where:

```math
d_{1,j} = \frac{\ln(S_0 / K_j) + \left(\mu - q + \frac{1}{2}\sigma^2\right)T}{\sigma \sqrt{T}}, \quad d_{2,j} = d_{1,j} - \sigma \sqrt{T}
```

To map percentile distributions, terminal prices are simulated via:

```math
S_T^{(i)} = S_0 \exp\left(\left(\mu - q - \frac{1}{2}\sigma^2\right)T + \sigma \sqrt{T} Z^{(i)}\right), \quad Z^{(i)} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)
```

---

### Model 2: Black-Scholes Mixture Model (Bootstrapped Empirical Volatility)

#### 1. Conceptual Intuition
Recognizes that volatility is not constant over time. Rather than assuming a single static baseline, this model repeatedly samples from historically realized rolling volatility regimes matching the duration of the trade. 

#### 2. Probabilistic Notation & Distributional Assumptions
Conditioned on a specific volatility state, log returns are locally Gaussian:

```math
\left.\ln\left(\frac{S_T}{S_0}\right) \right| \sigma_i \sim \mathcal{N}\left(\left(\mu - q - \frac{1}{2}\sigma_i^2\right)T, \, \sigma_i^2 T\right)
```

The marginal distribution of terminal returns across the entire sample space constitutes a continuous scale mixture of Gaussians:

```math
F(S_T) = \int_{0}^{\infty} \Phi\left(\frac{\ln(S_T/S_0) - \mu_{adj}(\sigma)T}{\sigma \sqrt{T}}\right) dF_{\sigma}(\sigma)
```

#### 3. Calibration & Parameter Estimation
For a trade with D trading days to expiration:

1. Compute the series of overlapping historical annualized standard deviations across the full price history:
   
```math
\sigma_{\tau} = \sqrt{\frac{252}{D-1} \sum_{k=\tau-D+1}^{\tau} (R_k - \bar{R}_{\tau})^2}
```

2. Compile the discrete empirical set:
   
```math
\Omega_{\sigma} = \{\sigma_D, \sigma_{D+1}, \dots, \sigma_N\}
```

#### 4. Valuation & Simulation Process
1. For each simulation path:
   * Draw a volatility parameter uniformly with replacement from the historical set.
   * Compute the analytical Black-Scholes structure value using the sampled volatility.
   * Generate terminal spot price:
     
```math
S_T^{(i)} = S_0 \exp\left((\mu - q - \frac{1}{2}\sigma_i^2)T + \sigma_i \sqrt{T} Z^{(i)}\right)
```
     
2. Calculate the discounted expected structural value:
   
```math
\mathbb{E}[V] = e^{-rT} \frac{1}{M} \sum_{i=1}^M V(\sigma_i)
```

---

### Model 3: Black-Scholes Hybrid Mixture Model (Empirical Body + EVT Volatility Tail)

#### 1. Conceptual Intuition
Model 2 suffers from **empirical truncation**: it cannot simulate a volatility regime higher than the historical sample maximum. Model 3 splices the empirical volatility distribution with a Generalized Pareto Distribution (GPD) fitted to the top 10% extreme volatility exceedances.

#### 2. Probabilistic Notation & Distributional Assumptions
Let the rolling volatility be a random variable. The tail distribution above the 90th percentile threshold (u) is modeled via the Pickands-Balkema-de Haan Theorem.

The 90th percentile threshold:
```math
u = F_{\Sigma}^{-1}(0.90)
```

The GPD formulation for the exceedances:
```math
\mathbb{P}(\Sigma - u \le y \mid \Sigma > u) \approx G_{\xi, \beta}(y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}
```

The composite cumulative distribution function for volatility is:

```math
F_{\text{Hybrid}}(\sigma) = \begin{cases} 
\hat{F}_n(\sigma) & \text{for } \sigma \le u \\
(1 - 0.10) + 0.10 \cdot G_{\xi, \beta}(\sigma - u) & \text{for } \sigma > u 
\end{cases}
```

#### 3. Calibration & Parameter Estimation
1. Compute the empirical 90th percentile threshold.
2. Extract historical exceedances.
3. Fit shape and scale via Maximum Likelihood Estimation (MLE):
   
```math
\mathcal{L}(\xi, \beta; Y) = -k \ln \beta - \left(1 + \frac{1}{\xi}\right) \sum_{j=1}^k \ln\left(1 + \frac{\xi y_j}{\beta}\right)
```

#### 4. Valuation & Simulation Process
1. Draw standard uniform random variate.
2. Apply Inverse Transform Sampling:
   
```math
\sigma_i = \begin{cases} 
\text{quantile of } \hat{F}_n \text{ at } U^{(i)} & \text{if } U^{(i)} \le 0.90 \\
u + \frac{\beta}{\xi}\left[\left(\frac{1 - U^{(i)}}{0.10}\right)^{-\xi} - 1\right] & \text{if } U^{(i)} > 0.90 
\end{cases}
```

3. Evaluate analytical option values at the sampled volatility and simulate paths.

---

### Model 4: Return-Based EVT Model (Empirical Center + EVT Tails)

#### 1. Conceptual Intuition
Abandons Black-Scholes analytical integration entirely. Simulates price paths step-by-step using a semi-parametric daily return distribution. Normal trading days are sampled directly from the empirical center, while the extreme left (crashes) and right (squeezes) tails are generated from parametric Generalized Pareto Distributions.

#### 2. Probabilistic Notation & Distributional Assumptions
The daily log return distribution is partitioned into three regimes:

```math
F_R(r) = \begin{cases} 
\tau_L \left[1 + \frac{\xi_L (|r| - |u_L|)}{\beta_L}\right]^{-1/\xi_L} & \text{for } r < u_L \quad (\text{Left Crash Tail}) \\
\hat{F}_{\text{emp}}(r) & \text{for } u_L \le r \le u_R \quad (\text{Empirical Body}) \\
1 - \tau_R \left[1 + \frac{\xi_R (r - u_R)}{\beta_R}\right]^{-1/\xi_R} & \text{for } r > u_R \quad (\text{Right Squeeze Tail}) 
\end{cases}
```

#### 3. Calibration & Parameter Estimation
1. Determine thresholds at the 5th and 95th percentiles.
2. Form exceedance vectors for left tail losses and right tail gains.
3. Maximize GPD log-likelihood to obtain shape and scale parameters.
4. Extract centered interior returns.

#### 4. Valuation & Simulation Process
For each simulation path:
1. For each discrete trading day:
   * Draw uniform variate.
   * If in the bottom 5%, draw a left-tail loss: 
```math
R_t = u_L - \text{GPD}^{-1}(U_t/0.05; \, \xi_L, \beta_L) - \bar{R}
```
   * If in the top 5%, draw a right-tail gain: 
```math
R_t = u_R + \text{GPD}^{-1}((U_t - 0.95)/0.05; \, \xi_R, \beta_R) - \bar{R}
```
   * Otherwise, sample randomly with replacement from the center.
2. Apply drift and compound terminal spot:
   
```math
S_T^{(i)} = S_0 \exp\left(\sum_{t=1}^D R_t^{(i)} + \left(\mu - q - \frac{1}{2}\sigma_{\text{hist}}^2\right) D \cdot \Delta t\right)
```

3. Price structure: 
   
```math
\mathbb{E}[V] = e^{-rT} \frac{1}{M} \sum_{i=1}^M \text{Payoff}(S_T^{(i)})
```

---

### Model 5: Declustered Return-Based EVT Model (Runs Method)

#### 1. Conceptual Intuition
Standard EVT requires returns to be independent and identically distributed. In financial markets, large price shocks cluster together in time. Fitting EVT directly to dependent raw returns violates asymptotic assumptions. Model 5 implements the **Runs Method** to cluster dependent exceedances, extracting only the single maximum shock per cluster to fit true, independent block maxima.

#### 2. Probabilistic Notation & Distributional Assumptions
Let an indicator sequence be defined for exceedances:

```math
I_t = \mathbb{I}_{\{R_t < u_L\}}, \quad t \in \{1, \dots, N\}
```

An exceedance cluster is defined such that events are separated by fewer than a specific number of days:

```math
t_{j+1} - t_j \le k
```

The cluster maximum loss is:

```math
\tilde{Y}_m = \max_{t \in C_m} (|R_t| - |u_L|)
```

The resulting declustered exceedances converge asymptotically to a generalized Pareto distribution.

#### 3. Calibration & Parameter Estimation
1. Scan historical returns chronologically. Group consecutive exceedances separated by fewer than 5 days into a unified cluster.
2. Isolate the extreme peak within each cluster:
   * Left tail: 
```math
\tilde{y}_{L, m} = |u_L| - \min_{t \in C_m}(R_t)
```
   * Right tail: 
```math
\tilde{y}_{R, m} = \max_{t \in C_m}(R_t) - u_R
```
3. Fit declustered GPD parameters using MLE exclusively on the independent peak vector.

#### 4. Valuation & Simulation Process
Path generation follows the daily step sampling protocol of Model 4, but draws tail innovations from the declustered GPD distributions. This prevents artificial amplification of tail risk caused by serially dependent crash runs.

---

### Model 6: GARCH-EVT Filtered Historical Simulation (L2 Norm)

#### 1. Conceptual Intuition
Combines parametric conditional heteroskedasticity with non-parametric EVT innovations. A large market shock today directly increases conditional volatility tomorrow. Model 6 fits a GARCH(1,1) model to historical returns, standardizes the returns into independent residuals, fits EVT to those residuals, and simulates paths where daily shocks recursively feed back into next-day volatility.

#### 2. Probabilistic Notation & Distributional Assumptions
The return process is decomposed into a conditional mean, conditional variance, and standardized innovation:

```math
R_t = \mu_t + \sigma_t Z_t
```

The conditional variance evolves according to a GARCH(1,1) process:

```math
\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2
```

The innovation distribution is modeled via EVT applied to the standardized residuals.

#### 3. Calibration & Parameter Estimation
1. Fit by minimizing the negative log-likelihood of centered returns:
   
```math
\mathcal{L}_{GARCH}(\omega, \alpha, \beta) = \frac{1}{2} \sum_{t=1}^N \left[ \ln(\sigma_t^2) + \frac{(R_t - \bar{R})^2}{\sigma_t^2} \right]
```

2. Extract standardized innovations:
   
```math
z_t = \frac{R_t - \bar{R}}{\sigma_t}
```

3. Fit GPD shape and scale parameters to the extreme percentiles of the standardized residuals.

#### 4. Valuation & Simulation Process
For each path:
1. Initialize variance at current market state.
2. For each day:
   * Draw an EVT-standardized shock.
   * Compute dynamic physical drift: 
```math
\mu_t = \left(\mu - q - \frac{1}{2}(\sigma_t^2 \cdot 252)\right) \Delta t
```
   * Calculate daily log return: 
```math
r_t = \mu_t + \sigma_t Z_t
```
   * Update spot price.
   * Update next-day conditional variance: 
```math
\sigma_{t+1}^2 = \omega + \alpha (\sigma_t Z_t)^2 + \beta \sigma_t^2
```
3. Compute discounted terminal payoffs across all simulated paths.

---

### Model 7: MAD-Based Filtered Historical Simulation (L1 Norm + EVT)

#### 1. Conceptual Intuition
Standard GARCH(1,1) relies on squared shocks (an L2 norm). For fat-tailed asset returns, the theoretical fourth moment is infinite. Squaring extreme returns destabilizes calibration, causing sample estimators to explode unreliably. Model 7 replaces the fragile L2 variance filter with an L1 Mean Absolute Deviation (MAD) Exponentially Weighted Moving Average (EWMA) filter. 

#### 2. Probabilistic Notation & Distributional Assumptions
The daily price process follows:

```math
R_t = \mu_t + \text{MAD}_t \cdot z_t
```

The L1 scale parameter updates via EWMA with decay factor:

```math
\text{MAD}_t = \lambda \text{MAD}_{t-1} + (1 - \lambda)|R_{t-1}|
```

Standardized residuals are declustered via the Runs Method to isolate independent shock peaks, and tail exceedances are fitted to a GPD.

#### 3. Calibration & Parameter Estimation
1. Compute the historical recursive MAD series:
   
```math
\text{MAD}_1 = \frac{1}{N}\sum_{k=1}^N |R_k|
```
```math
\text{MAD}_t = \lambda \text{MAD}_{t-1} + (1 - \lambda)|R_{t-1}|
```

2. Extract L1 standardized residuals:
   
```math
z_t = \frac{R_t}{\text{MAD}_t}
```

3. Establish thresholds, apply the Runs Method, and fit GPD parameters to residual exceedances via MLE.

#### 4. Valuation & Simulation Process
For each path:
1. Initialize variance at current market state:
   
```math
\text{MAD}_1 = \text{MAD}_{\text{latest}}, \quad S_1 = S_0
```

2. For each day:
   * Draw an EVT residual from the spliced declustered residual distribution.
   * Compute the simulated physical return component: 
```math
r_{\text{sim}} = z_t \cdot \text{MAD}_t
```
   * Calculate the local drift using the calibrated annualized scale correction:
     
```math
\mu_t = \left(\mu - q - \frac{1}{2}(\text{MAD}_t^2 \cdot 252)\right) \Delta t
```
   
   * Update underlying price.
   * Update the L1 scale parameter recursively for tomorrow:
     
```math
\text{MAD}_{t+1} = \lambda \text{MAD}_t + (1 - \lambda)|r_{\text{sim}}|
```

3. Discount the terminal intrinsic payoffs back to present value.

---

## 🔬 Mathematical Parameters & Statistical Interpretations

| Parameter | Mathematical Domain | Theoretical Description |
| :--- | :--- | :--- |
| **Tail Shape ($\xi$)** | $\xi \in (-\infty, \infty)$ | Governs tail decay rate. $\xi \le 0$ implies thin/Gaussian decay. $0 < \xi < 0.5$ indicates heavy, Paretian tails. $\xi \ge 0.5$ implies infinite variance ($\mathbb{E}[X^2] = \infty$). Tail index is given by $\alpha_{\text{tail}} = 1/\xi$. |
| **Tail Scale ($\beta$)** | $\beta > 0$ | Measures the dispersion and magnitude of exceedances above the threshold. |
| **GARCH $\omega$** | $\omega > 0$ | The long-term baseline variance intercept. |
| **GARCH $\alpha$** | $\alpha \ge 0$ | The shock reaction coefficient (sensitivity of conditional variance to the previous day's squared innovation). |
| **GARCH $\beta$** | $\beta \ge 0$ | The persistence coefficient (memory of past volatility regimes). |
| **Persistence ($\alpha+\beta$)** | $\alpha + \beta < 1$ | The rate of mean-reversion toward long-term variance. Values $> 0.98$ represent severe volatility clustering. |
| **MAD Decay ($\lambda$)** | $\lambda \in (0, 1)$ | Controls the exponential memory horizon of the L1 MAD filter. Set to $\lambda = 0.94$ (effective memory horizon of $\sim \frac{1}{1-\lambda} \approx 17$ trading days). |
| **Runs Window ($k$)** | $k \in \mathbb{N}^+$ | Separation window for the Runs Method declustering algorithm ($k=5$ trading days). Eliminates dependent intra-cluster peaks. |

---

## 📊 Analytical Greeks & Risk Profiling Engine

The application includes an analytical Greek calculation engine supporting first-order, second-order, and cross-sensitivities across multi-leg structures:

### First-Order Greeks
* **Delta ($\Delta$):** First derivative of structure value with respect to spot price:
  
```math
\Delta = \frac{\partial V}{\partial S} = \sum_{j=1}^M w_j \phi_j e^{-q T} \mathcal{N}(\phi_j d_{1,j})
```

* **Vega ($\nu$):** Sensitivity to a 1% absolute move in implied volatility:
  
```math
\nu = \frac{1}{100} \frac{\partial V}{\partial \sigma} = \frac{1}{100} \sum_{j=1}^M w_j S_0 e^{-q T} \phi_j n(d_{1,j}) \sqrt{T}
```

* **Theta ($\Theta$):** Calendar day time decay:
  
```math
\Theta_{\text{call}} = \frac{1}{365}\left[ -\frac{S_0 n(d_1)\sigma e^{-q T}}{2\sqrt{T}} - r K e^{-r T} \mathcal{N}(d_2) + q S_0 e^{-q T} \mathcal{N}(d_1) \right]
```
  
```math
\Theta_{\text{put}} = \frac{1}{365}\left[ -\frac{S_0 n(d_1)\sigma e^{-q T}}{2\sqrt{T}} + r K e^{-r T} \mathcal{N}(-d_2) - q S_0 e^{-q T} \mathcal{N}(-d_1) \right]
```

* **Rho ($\rho$):** Sensitivity to a 1% change in the risk-free interest rate:
  
```math
\rho = \frac{1}{100} \frac{\partial V}{\partial r} = \frac{1}{100} \sum_{j=1}^M w_j \phi_j K_j T e^{-r T} \mathcal{N}(\phi_j d_{2,j})
```

### Second-Order & Cross Greeks
* **Gamma ($\Gamma$):** Curvature and directional acceleration:
  
```math
\Gamma = \frac{\partial^2 V}{\partial S^2} = \sum_{j=1}^M w_j \frac{e^{-q T} n(d_{1,j})}{S_0 \sigma \sqrt{T}}
```

* **Vomma (Volga):** Volatility convexity (rate of change of Vega with respect to volatility):
  
```math
\text{Vomma} = \frac{\partial^2 V}{\partial \sigma^2} = \nu \cdot \frac{d_{1} d_{2}}{\sigma}
```

* **Vanna:** Cross-derivative of option value with respect to spot price and volatility:
  
```math
\text{Vanna} = \frac{\partial^2 V}{\partial S \partial \sigma} = -e^{-q T} n(d_1) \frac{d_2}{\sigma}
```

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+
* Git

### Local Deployment
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/option-valuation-tool.git
   cd option-valuation-tool
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch the Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```

---

## 📦 Dependencies
* **Streamlit:** Web interface architecture and interactive state management.
* **NumPy:** Vectorized Monte Carlo trajectory simulations and numerical linear algebra.
* **Pandas:** Time series processing and business holiday calendar math.
* **SciPy:** Maximum Likelihood Estimation optimization and Extreme Value distributions.
* **Plotly:** Interactive terminal payoff charts and 3D risk surfaces.
* **yfinance:** Market data extraction with underlying proxy mapping.