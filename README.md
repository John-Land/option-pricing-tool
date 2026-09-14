# Quantitative Option Valuation & Fat-Tail Risk Analysis Tool

A high-performance quantitative finance and options engineering application built with Streamlit, Plotly, SciPy, and NumPy. 

This platform replaces the Gaussian and constant-volatility assumptions of classical financial engineering with non-linear volatility dynamics and Extreme Value Theory (EVT). It evaluates single-leg options and complex multi-leg combinations across physical real-world probability distributions $\mathbb{P}$ rather than artificial risk-neutral surfaces $\mathbb{Q}$, capturing volatility clustering, asymmetric tail decay, parameter estimation risk, and Jensen's inequality convexity.

---

## 📐 Theoretical Framework: Physical Valuation vs. Risk-Neutral Pricing

Standard option pricing models operate under the risk-neutral measure $\mathbb{Q}$, discounting expected terminal payoffs at the risk-free rate $r$ under the assumption that all underlying assets drift at $r - q$. While mathematically convenient for dynamic delta-hedging in complete markets, $\mathbb{Q}$-pricing fails to reflect:

1. **Physical Expectation $\mathbb{P}$:** Real-world assets compound with an equity risk premium (ERP) and idiosyncratic alpha: 
   
   $$\mu = r + \beta(\text{ERP}) + \alpha_{\text{drift}}/T$$

2. **Paretian Tail Power Laws:** Equity crash distributions exhibit heavy tails characterized by Generalized Pareto Distributions where extreme outcomes decay as power laws $P(X > x) \sim x^{-\alpha}$, rather than exponentially as in Gaussian models.

3. **Volatility Convexity (Jensen's Inequality):** Because option contracts possess positive Vega and Vomma ($\frac{\partial^2 V}{\partial \sigma^2} > 0$), integrating over an uncertain or fluctuating volatility distribution yields an expected value strictly greater than evaluating the contract at the mean volatility:
   
   $$\mathbb{E}[V(\sigma)] \ge V(\mathbb{E}[\sigma])$$

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
Values the option structure analytically under the classical Black-Scholes-Merton assumption that the underlying asset follows continuous Geometric Brownian Motion (GBM) with constant volatility $\sigma$ and constant drift. It ignores fat tails, skew, and market shocks.

#### 2. Probabilistic Notation & Distributional Assumptions
Under the physical probability measure $\mathbb{P}$, the continuous price process $S_t$ satisfies the Stochastic Differential Equation (SDE):

$$dS_t = (\mu - q) S_t \, dt + \sigma S_t \, dW_t$$

where $W_t$ is a standard Brownian motion under $\mathbb{P}$, $\mu$ is the subjective physical drift, and $q$ is the continuous dividend yield.

By Itô's Lemma, the log return over horizon $T$ follows a Gaussian distribution:

$$\ln\left(\frac{S_T}{S_0}\right) \sim \mathcal{N}\left(\left(\mu - q - \frac{1}{2}\sigma^2\right)T, \, \sigma^2 T\right)$$

The terminal price $S_T$ is distributed lognormally:

$$S_T \sim \text{Lognormal}\left(\ln S_0 + \mu_{adj} T, \, \sigma^2 T\right), \quad \text{where } \mu_{adj} = \mu - q - \frac{1}{2}\sigma^2$$

#### 3. Calibration & Parameter Estimation
* **Volatility ($\sigma$):** Sample standard deviation of historical daily log returns scaled to an annual horizon:
  
  $$\sigma = \sqrt{\frac{252}{N-1} \sum_{k=1}^N (R_k - \bar{R})^2}$$

* **Physical Drift ($\mu$):** User-calibrated Capital Asset Pricing Model (CAPM) structure:
  
  $$\mu = r + \beta \cdot \text{ERP} + \frac{\alpha_{\text{drift}}}{T}$$

#### 4. Valuation & Simulation Process
The terminal intrinsic value for an arbitrary set of call/put legs $j \in \{1, \dots, M\}$ with strike $K_j$, direction indicator $\phi_j \in \{+1 \text{ (call)}, -1 \text{ (put)}\}$, and position weight $w_j$ is evaluated analytically:

$$V_0 = e^{-rT} \sum_{j=1}^M w_j \left[ \phi_j S_0 e^{(\mu - q)T} \mathcal{N}(\phi_j d_{1,j}) - \phi_j K_j \mathcal{N}(\phi_j d_{2,j}) \right]$$

where:

$$d_{1,j} = \frac{\ln(S_0 / K_j) + \left(\mu - q + \frac{1}{2}\sigma^2\right)T}{\sigma \sqrt{T}}, \quad d_{2,j} = d_{1,j} - \sigma \sqrt{T}$$

To map percentile distributions, terminal prices are simulated via:

$$S_T^{(i)} = S_0 \exp\left(\left(\mu - q - \frac{1}{2}\sigma^2\right)T + \sigma \sqrt{T} Z^{(i)}\right), \quad Z^{(i)} \stackrel{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$$

---

### Model 2: Black-Scholes Mixture Model (Bootstrapped Empirical Volatility)

#### 1. Conceptual Intuition
Recognizes that volatility is not constant over time. Rather than assuming a single static $\sigma$, this model repeatedly samples from historically realized rolling volatility regimes matching the duration of the trade ($N$ trading days to expiry). 

#### 2. Probabilistic Notation & Distributional Assumptions
Conditioned on a specific volatility state $\sigma_i$, log returns are locally Gaussian:

$$\left.\ln\left(\frac{S_T}{S_0}\right) \right| \sigma_i \sim \mathcal{N}\left(\left(\mu - q - \frac{1}{2}\sigma_i^2\right)T, \, \sigma_i^2 T\right)$$

The marginal distribution of terminal returns across the entire sample space constitutes a continuous scale mixture of Gaussians:

$$F(S_T) = \int_{0}^{\infty} \Phi\left(\frac{\ln(S_T/S_0) - \mu_{adj}(\sigma)T}{\sigma \sqrt{T}}\right) dF_{\sigma}(\sigma)$$

where $F_{\sigma}$ is the empirical cumulative distribution function (ECDF) of rolling historical volatility.

#### 3. Calibration & Parameter Estimation
For a trade with $D$ trading days to expiration:

1. Compute the series of overlapping $D$-day historical annualized standard deviations across the full price history:
   
   $$\sigma_{\tau} = \sqrt{\frac{252}{D-1} \sum_{k=\tau-D+1}^{\tau} (R_k - \bar{R}_{\tau})^2}, \quad \tau \in [D, N]$$

2. Compile the discrete empirical set $\Omega_{\sigma} = \{\sigma_D, \sigma_{D+1}, \dots, \sigma_N\}$.

#### 4. Valuation & Simulation Process
1. For each simulation path $i \in \{1, \dots, N_{paths}\}$:
   * Draw a volatility parameter uniformly with replacement: $\sigma_i \sim \text{Uniform}(\Omega_{\sigma})$.
   * Compute the analytical Black-Scholes structure value $V(\sigma_i)$ using sample volatility $\sigma_i$.
   * Generate terminal spot price $S_T^{(i)} = S_0 \exp\left((\mu - q - \frac{1}{2}\sigma_i^2)T + \sigma_i \sqrt{T} Z^{(i)}\right)$, where $Z^{(i)} \sim \mathcal{N}(0,1)$.
2. Calculate the discounted expected structural value:
   
   $$\mathbb{E}[V] = e^{-rT} \frac{1}{M} \sum_{i=1}^M V(\sigma_i)$$

---

### Model 3: Black-Scholes Hybrid Mixture Model (Empirical Body + EVT Volatility Tail)

#### 1. Conceptual Intuition
Model 2 suffers from **empirical truncation**: it cannot simulate a volatility regime higher than the historical sample maximum. If an asset has never experienced a severe market dislocation within its specific data window, Model 2 understates extreme tail convexity. Model 3 splices the empirical volatility distribution with a Generalized Pareto Distribution (GPD) fitted to the top 10% extreme volatility exceedances.

#### 2. Probabilistic Notation & Distributional Assumptions
Let $\Sigma$ denote the rolling volatility random variable. The tail distribution above threshold $u$ (the 90th percentile $u = F_{\Sigma}^{-1}(0.90)$) is modeled via the Pickands-Balkema-de Haan Theorem:

$$\mathbb{P}(\Sigma - u \le y \mid \Sigma > u) \approx G_{\xi, \beta}(y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}$$

where:
* $\xi \in \mathbb{R}$ is the GPD shape parameter (tail index $\alpha_{\text{tail}} = 1/\xi$).
* $\beta > 0$ is the scale parameter.
* Support is $y \ge 0$ for $\xi \ge 0$, and $0 \le y \le -\beta/\xi$ for $\xi < 0$.

The composite cumulative distribution function for volatility is:

$$F_{\text{Hybrid}}(\sigma) = \begin{cases} \hat{F}_n(\sigma) & \text{for } \sigma \le u \\ (1 - 0.10) + 0.10 \cdot G_{\xi, \beta}(\sigma - u) & \text{for } \sigma > u \end{cases}$$

#### 3. Calibration & Parameter Estimation
1. Compute the empirical 90th percentile threshold: $u = \text{Percentile}(\Omega_{\sigma}, 90)$.
2. Extract historical exceedances: $Y = \{\sigma - u \mid \sigma \in \Omega_{\sigma}, \, \sigma > u\}$.
3. Fit shape ($\xi$) and scale ($\beta$) via Maximum Likelihood Estimation (MLE):
   
   $$\mathcal{L}(\xi, \beta; Y) = -k \ln \beta - \left(1 + \frac{1}{\xi}\right) \sum_{j=1}^k \ln\left(1 + \frac{\xi y_j}{\beta}\right)$$

#### 4. Valuation & Simulation Process
1. Draw standard uniform random variate $U^{(i)} \sim \mathcal{U}(0, 1)$.
2. Apply Inverse Transform Sampling:
   
   $$\sigma_i = \begin{cases} \text{quantile of } \hat{F}_n \text{ at } U^{(i)} & \text{if } U^{(i)} \le 0.90 \\ u + \frac{\beta}{\xi}\left[\left(\frac{1 - U^{(i)}}{0.10}\right)^{-\xi} - 1\right] & \text{if } U^{(i)} > 0.90 \end{cases}$$

3. Evaluate analytical option values at $\sigma_i$ and simulate $S_T^{(i)}$ paths using sampled $\sigma_i$.

---

### Model 4: Return-Based EVT Model (Empirical Center + EVT Tails)

#### 1. Conceptual Intuition
Abandons Black-Scholes analytical integration entirely. Simulates price paths step-by-step using a semi-parametric daily return distribution. Normal trading days are sampled directly from the empirical center, while the extreme left (crashes) and right (squeezes) tails are generated from parametric Generalized Pareto Distributions fitted to the historical 5th and 95th percentiles.

#### 2. Probabilistic Notation & Distributional Assumptions
Let $R_t = \ln(S_t / S_{t-1})$ denote daily log returns. The distribution $F_R(r)$ is partitioned into three regimes:

$$F_R(r) = \begin{cases} \tau_L \left[1 + \frac{\xi_L (|r| - |u_L|)}{\beta_L}\right]^{-1/\xi_L} & \text{for } r < u_L \quad (\text{Left Crash Tail}) \\ \hat{F}_{\text{emp}}(r) & \text{for } u_L \le r \le u_R \quad (\text{Empirical Body}) \\ 1 - \tau_R \left[1 + \frac{\xi_R (r - u_R)}{\beta_R}\right]^{-1/\xi_R} & \text{for } r > u_R \quad (\text{Right Squeeze Tail}) \end{cases}$$

where $u_L$ is the 5th percentile return, $u_R$ is the 95th percentile return, and $\tau_L = \tau_R = 0.05$.

#### 3. Calibration & Parameter Estimation
1. Determine thresholds: $u_L = \text{Percentile}(R, 5)$ and $u_R = \text{Percentile}(R, 95)$.
2. Form exceedance vectors:
   * Left tail losses: $Y_L = |r| - |u_L|$ for all $r < u_L$.
   * Right tail gains: $Y_R = r - u_R$ for all $r > u_R$.
3. Maximize GPD log-likelihood to obtain $(\xi_L, \beta_L)$ and $(\xi_R, \beta_R)$.
4. Extract centered interior returns: $R_{\text{center}} = \{r - \bar{R} \mid u_L \le r \le u_R\}$.

#### 4. Valuation & Simulation Process
For each simulation path $i$:
1. For each discrete trading day $t \in \{1, \dots, D\}$:
   * Draw uniform variate $U_t \sim \mathcal{U}(0, 1)$.
   * If $U_t < 0.05$, draw a left-tail loss: $R_t = u_L - \text{GPD}^{-1}(U_t/0.05; \, \xi_L, \beta_L) - \bar{R}$.
   * If $U_t > 0.95$, draw a right-tail gain: $R_t = u_R + \text{GPD}^{-1}((U_t - 0.95)/0.05; \, \xi_R, \beta_R) - \bar{R}$.
   * Otherwise, sample randomly with replacement from $R_{\text{center}}$.
2. Apply drift and compound terminal spot:
   
   $$S_T^{(i)} = S_0 \exp\left(\sum_{t=1}^D R_t^{(i)} + \left(\mu - q - \frac{1}{2}\sigma_{\text{hist}}^2\right) D \cdot \Delta t\right), \quad \Delta t = \frac{1}{252}$$

3. Price structure: 
   
   $$\mathbb{E}[V] = e^{-rT} \frac{1}{M} \sum_{i=1}^M \text{Payoff}(S_T^{(i)})$$

---

### Model 5: Declustered Return-Based EVT Model (Runs Method)

#### 1. Conceptual Intuition
Standard EVT requires returns to be independent and identically distributed (i.i.d.). In financial markets, large price shocks cluster together in time (volatility clustering). Fitting EVT directly to dependent raw returns violates the asymptotic assumptions of the Fisher-Tippett-Gnedenko theorem, overstating the frequency of independent black swans. 

Model 5 implements the **Runs Method** ($k=5$ day window) to cluster dependent exceedances, extracting only the single maximum shock per cluster to fit true, independent block maxima.

#### 2. Probabilistic Notation & Distributional Assumptions
Let $\{I_t\}_{t=1}^N$ be an indicator sequence where $I_t = \mathbb{I}_{\{R_t < u_L\}}$. An exceedance cluster $C_m = \{R_{t_1}, R_{t_2}, \dots, R_{t_p}\}$ is defined such that:

$$t_{j+1} - t_j \le k, \quad \forall j \in \{1, \dots, p-1\}$$

where $k=5$ trading days defines the cluster run length. The cluster maximum loss is:

$$\tilde{Y}_m = \max_{t \in C_m} (|R_t| - |u_L|)$$

The resulting declustered exceedances $\{\tilde{Y}_m\}_{m=1}^{N_c}$ converge asymptotically to a generalized Pareto distribution under the Extremal Index $\theta \in (0, 1]$:

$$\mathbb{P}(\tilde{Y} \le y) = 1 - \left(1 + \frac{\xi_D y}{\beta_D}\right)^{-1/\xi_D}$$

#### 3. Calibration & Parameter Estimation
1. Scan historical returns $R_t$ chronologically. Group consecutive exceedances separated by fewer than $k=5$ days into a unified cluster.
2. Isolate the extreme peak within each cluster:
   * Left tail: $\tilde{y}_{L, m} = |u_L| - \min_{t \in C_m}(R_t)$
   * Right tail: $\tilde{y}_{R, m} = \max_{t \in C_m}(R_t) - u_R$
3. Fit declustered GPD parameters $(\xi_{L, \text{dec}}, \beta_{L, \text{dec}})$ and $(\xi_{R, \text{dec}}, \beta_{R, \text{dec}})$ using MLE exclusively on the independent peak vector.

#### 4. Valuation & Simulation Process
Path generation follows the daily step sampling protocol of Model 4, but draws tail innovations from the declustered GPD distributions $(\xi_{\text{dec}}, \beta_{\text{dec}})$. This prevents artificial amplification of tail risk caused by serially dependent crash runs.

---

### Model 6: GARCH-EVT Filtered Historical Simulation ($L_2$ Norm)

#### 1. Conceptual Intuition
Combines parametric conditional heteroskedasticity with non-parametric EVT innovations. Standard EVT assumes stationary variance. In reality, a large market shock today directly increases conditional volatility tomorrow. 

Model 6 fits a GARCH(1,1) model to historical returns, standardizes the returns into independent identically distributed (i.i.d.) residuals, fits EVT to those standardized residuals, and simulates paths where daily shocks recursively feed back into next-day volatility.

#### 2. Probabilistic Notation & Distributional Assumptions
The return process is decomposed into a conditional mean, conditional variance, and standardized innovation:

$$R_t = \mu_t + \sigma_t Z_t, \quad Z_t \stackrel{\text{i.i.d.}}{\sim} F_Z(0, 1)$$

The conditional variance $\sigma_t^2$ evolves according to a GARCH(1,1) process ($L_2$ norm):

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

where $\epsilon_{t-1} = R_{t-1} - \bar{R}$ is the demeaned return shock. Stationarity requires $\omega > 0, \, \alpha \ge 0, \, \beta \ge 0$, and $\alpha + \beta < 1$.

The innovation distribution $F_Z(z)$ is modeled via EVT:

$$F_Z(z) = \begin{cases} \tau_L \left[1 + \frac{\xi_L (|z| - |u_L|)}{\beta_L}\right]^{-1/\xi_L} & \text{for } z < u_L \\ \hat{F}_{\text{emp}}(z) & \text{for } u_L \le z \le u_R \\ 1 - \tau_R \left[1 + \frac{\xi_R (z - u_R)}{\beta_R}\right]^{-1/\xi_R} & \text{for } z > u_R \end{cases}$$

#### 3. Calibration & Parameter Estimation
1. Fit $(\omega, \alpha, \beta)$ by minimizing the negative log-likelihood of centered returns:
   
   $$\mathcal{L}_{GARCH}(\omega, \alpha, \beta) = \frac{1}{2} \sum_{t=1}^N \left[ \ln(\sigma_t^2) + \frac{(R_t - \bar{R})^2}{\sigma_t^2} \right]$$

2. Extract standardized innovations:
   
   $$z_t = \frac{R_t - \bar{R}}{\sigma_t}, \quad t \in [1, N]$$

3. Fit GPD shape and scale parameters to the extreme 5th and 95th percentiles of the standardized residuals $\{z_t\}$.

#### 4. Valuation & Simulation Process
For each path $i$:
1. Initialize variance at current market state: $\sigma_1^2 = \sigma_{\text{latest}}^2$, and $S_1 = S_0$.
2. For each day $t \in \{1, \dots, D\}$:
   * Draw an EVT-standardized shock $Z_t$ using Inverse Transform Sampling.
   * Compute dynamic physical drift: $\mu_t = \left(\mu - q - \frac{1}{2}(\sigma_t^2 \cdot 252)\right) \Delta t$.
   * Calculate daily log return: $r_t = \mu_t + \sigma_t Z_t$.
   * Update spot price: $S_{t+1} = S_t \exp(r_t)$.
   * Update next-day conditional variance: $\sigma_{t+1}^2 = \omega + \alpha (\sigma_t Z_t)^2 + \beta \sigma_t^2$.
3. Compute discounted terminal payoffs across all simulated paths.

---

### Model 7: MAD-Based Filtered Historical Simulation ($L_1$ Norm + EVT)

#### 1. Conceptual Intuition
Standard GARCH(1,1) relies on squared shocks ($\epsilon_{t-1}^2$, an $L_2$ norm). For fat-tailed asset returns where the tail index $\alpha \le 4$, the theoretical fourth moment is infinite:

$$\mathbb{E}[R^4] = \infty$$

Under these conditions, squaring extreme returns destabilizes calibration, causing sample variance estimators to explode unreliably. 

Model 7 replaces the fragile $L_2$ variance filter with an $L_1$ Mean Absolute Deviation (MAD) Exponentially Weighted Moving Average (EWMA) filter. Because the first moment of financial returns is always finite ($\alpha > 1$), standardizing returns by the $L_1$ norm yields robust residuals that are declustered and modeled via EVT without numerical instability.

#### 2. Probabilistic Notation & Distributional Assumptions
The daily price process follows:

$$R_t = \mu_t + \text{MAD}_t \cdot z_t, \quad z_t \stackrel{\text{i.i.d.}}{\sim} F_{\text{MAD-EVT}}$$

The $L_1$ scale parameter $\text{MAD}_t = \mathbb{E}[|R_t - \mathbb{E}[R_t]|]$ updates via EWMA with decay factor $\lambda = 0.94$:

$$\text{MAD}_t = \lambda \text{MAD}_{t-1} + (1 - \lambda)|R_{t-1}|$$

Standardized residuals $z_t = \frac{R_t}{\text{MAD}_t}$ are declustered via the Runs Method ($k=5$) to isolate independent shock peaks, and tail exceedances are fitted to a Generalized Pareto Distribution.

#### 3. Calibration & Parameter Estimation
1. Compute the historical recursive MAD series:
   
   $$\text{MAD}_1 = \frac{1}{N}\sum_{k=1}^N |R_k|, \quad \text{MAD}_t = \lambda \text{MAD}_{t-1} + (1 - \lambda)|R_{t-1}|$$

2. Extract $L_1$ standardized residuals:
   
   $$z_t = \frac{R_t}{\text{MAD}_t}$$

3. Establish 5th and 95th percentile thresholds on $\{z_t\}$. Apply the Runs Method ($k=5$) to extract declustered residual exceedances.
4. Fit GPD parameters $(\xi_L, \beta_L)$ and $(\xi_R, \beta_R)$ to residual exceedances via MLE.

#### 4. Valuation & Simulation Process
For each path $i$:
1. Initialize $\text{MAD}_1 = \text{MAD}_{\text{latest}}$, and $S_1 = S_0$.
2. For each day $t \in \{1, \dots, D\}$:
   * Draw an EVT residual $z_t$ from the spliced declustered residual distribution.
   * Compute the simulated physical return component: $r_{\text{sim}} = z_t \cdot \text{MAD}_t$.
   * Calculate the local drift using the calibrated annualized scale correction:
     
     $$\mu_t = \left(\mu - q - \frac{1}{2}(\text{MAD}_t^2 \cdot 252)\right) \Delta t$$
   
   * Update underlying price: $S_{t+1} = S_t \exp(r_{\text{sim}} + \mu_t)$.
   * Update the $L_1$ scale parameter recursively for tomorrow:
     
     $$\text{MAD}_{t+1} = \lambda \text{MAD}_t + (1 - \lambda)|r_{\text{sim}}|$$

3. Discount the terminal intrinsic payoffs back to $t_0$:
   
   $$\mathbb{E}[V] = e^{-rT} \frac{1}{M} \sum_{i=1}^M \text{Payoff}(S_T^{(i)})$$

---

## 🔬 Mathematical Parameters & Statistical Interpretations

| Parameter | Mathematical Domain | Theoretical Description |
| :--- | :--- | :--- |
| **Tail Shape ($\xi$)** | $\xi \in (-\infty, \infty)$ | Governs tail decay rate. $\xi \le 0$ implies thin/Gaussian decay. $0 < \xi < 0.5$ indicates heavy, Paretian tails. $\xi \ge 0.5$ implies infinite variance ($\mathbb{E}[X^2] = \infty$). Tail index is given by $\alpha_{\text{tail}} = 1/\xi$. |
| **Tail Scale ($\beta$)** | $\beta > 0$ | Measures the dispersion and magnitude of exceedances above the threshold $u$. |
| **GARCH $\omega$** | $\omega > 0$ | The long-term baseline variance intercept: $\sigma^2_{\infty} = \frac{\omega}{1 - (\alpha + \beta)}$. |
| **GARCH $\alpha$** | $\alpha \ge 0$ | The shock reaction coefficient (sensitivity of conditional variance to the previous day's squared innovation $\epsilon_{t-1}^2$). |
| **GARCH $\beta$** | $\beta \ge 0$ | The persistence coefficient (memory of past volatility regimes). |
| **Persistence ($\alpha+\beta$)** | $\alpha + \beta < 1$ | The rate of mean-reversion toward long-term variance. Values $> 0.98$ represent severe volatility clustering. |
| **MAD Decay ($\lambda$)** | $\lambda \in (0, 1)$ | Controls the exponential memory horizon of the $L_1$ MAD filter. Set to $\lambda = 0.94$ (effective memory horizon of $\sim \frac{1}{1-\lambda} \approx 17$ trading days). |
| **Runs Window ($k$)** | $k \in \mathbb{N}^+$ | Separation window for the Runs Method declustering algorithm ($k=5$ trading days). Eliminates dependent intra-cluster peaks. |

---

## 📊 Analytical Greeks & Risk Profiling Engine

The application includes an analytical Greek calculation engine supporting first-order, second-order, and cross-sensitivities across multi-leg structures:

### First-Order Greeks
* **Delta ($\Delta$):** First derivative of structure value with respect to spot price:
  
  $$\Delta = \frac{\partial V}{\partial S} = \sum_{j=1}^M w_j \phi_j e^{-q T} \mathcal{N}(\phi_j d_{1,j})$$

* **Vega ($\nu$):** Sensitivity to a $1\%$ absolute move in implied volatility:
  
  $$\nu = \frac{1}{100} \frac{\partial V}{\partial \sigma} = \frac{1}{100} \sum_{j=1}^M w_j S_0 e^{-q T} \phi_j n(d_{1,j}) \sqrt{T}$$

* **Theta ($\Theta$):** Calendar day time decay ($\frac{1}{365} \frac{\partial V}{\partial t}$):
  
  $$\Theta_{\text{call}} = \frac{1}{365}\left[ -\frac{S_0 n(d_1)\sigma e^{-q T}}{2\sqrt{T}} - r K e^{-r T} \mathcal{N}(d_2) + q S_0 e^{-q T} \mathcal{N}(d_1) \right]$$
  
  $$\Theta_{\text{put}} = \frac{1}{365}\left[ -\frac{S_0 n(d_1)\sigma e^{-q T}}{2\sqrt{T}} + r K e^{-r T} \mathcal{N}(-d_2) - q S_0 e^{-q T} \mathcal{N}(-d_1) \right]$$

* **Rho ($\rho$):** Sensitivity to a $1\%$ change in the risk-free interest rate:
  
  $$\rho = \frac{1}{100} \frac{\partial V}{\partial r} = \frac{1}{100} \sum_{j=1}^M w_j \phi_j K_j T e^{-r T} \mathcal{N}(\phi_j d_{2,j})$$

### Second-Order & Cross Greeks
* **Gamma ($\Gamma$):** Curvature and directional acceleration:
  
  $$\Gamma = \frac{\partial^2 V}{\partial S^2} = \sum_{j=1}^M w_j \frac{e^{-q T} n(d_{1,j})}{S_0 \sigma \sqrt{T}}$$

* **Vomma (Volga):** Volatility convexity (rate of change of Vega with respect to volatility):
  
  $$\text{Vomma} = \frac{\partial^2 V}{\partial \sigma^2} = \nu \cdot \frac{d_{1} d_{2}}{\sigma}$$

* **Vanna:** Cross-derivative of option value with respect to spot price and volatility:
  
  $$\text{Vanna} = \frac{\partial^2 V}{\partial S \partial \sigma} = -e^{-q T} n(d_1) \frac{d_2}{\sigma}$$

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+
* Git

### Local Deployment
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/option-valuation-tool.git](https://github.com/yourusername/option-valuation-tool.git)
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
* **SciPy:** Maximum Likelihood Estimation optimization (`scipy.optimize.minimize`) and Extreme Value distributions (`scipy.stats.genpareto`, `scipy.stats.norm`).
* **Plotly:** Interactive terminal payoff charts and 3D risk surfaces.
* **yfinance:** Market data extraction with underlying proxy mapping (e.g., SPY $\to$ ^GSPC back to 1927).