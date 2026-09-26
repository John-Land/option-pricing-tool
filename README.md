# Quantitative Option Valuation & Fat-Tail Risk Analysis Tool

A high-performance quantitative finance and options engineering application built with Streamlit, Plotly, SciPy, and NumPy. 

This platform replaces the Gaussian and constant-volatility assumptions of classical financial engineering with non-linear volatility dynamics and Extreme Value Theory (EVT). It evaluates single-leg options and complex multi-leg combinations across physical real-world probability distributions (P) rather than artificial risk-neutral surfaces (Q), capturing volatility clustering, asymmetric tail decay, parameter estimation risk, and Jensen's inequality convexity.

---

## 📐 Theoretical Framework: Physical Valuation vs. Risk-Neutral Pricing

Standard option pricing models operate under the risk-neutral measure, discounting expected terminal payoffs at the risk-free rate under the assumption that all underlying assets drift at the risk free rate minus dividend yield. While mathematically convenient for dynamic delta-hedging in complete markets, risk-neutral pricing fails to reflect:

1. **Physical Expectation:** Real-world assets compound with an equity risk premium (ERP) and idiosyncratic alpha: 
   
$$
\mu = r + \beta(\text{ERP}) + \alpha_{\text{drift}}/T
$$

2. **Paretian Tail Power Laws:** Equity crash distributions exhibit heavy tails characterized by Generalized Pareto Distributions where extreme outcomes decay as power laws rather than exponentially as in Gaussian models.

3. **Volatility Convexity (Jensen's Inequality):** Because option contracts possess positive Vega and Vomma, integrating over an uncertain or fluctuating volatility distribution yields an expected value strictly greater than evaluating the contract at the mean volatility:
   
$$
\mathbb{E}[V(\sigma)] \ge V(\mathbb{E}[\sigma])
$$

---

## 🧠 Comprehensive Model Hierarchy & Mathematical Formulations

The tool implements an evolutionary hierarchy of seven valuation engines. Each engine progressively strips away artificial constraints to model real-world asset price dynamics.

~~~text
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
~~~

---

### 1. Standard Black-Scholes (Analytical Constant Volatility)

**Intuitive Foundation**
The baseline Black-Scholes-Merton model establishes the theoretical framework by assuming the underlying asset price follows continuous Geometric Brownian Motion (GBM). It assumes market volatility is perfectly constant and that price returns are continuously compounded and symmetrically distributed. It explicitly ignores fat tails, volatility clustering, and the parameter uncertainty of extreme shocks.

**Distributional Assumptions**
Under the physical probability measure $\mathbb{P}$, the continuous price process satisfies the Stochastic Differential Equation (SDE):

$$
dS_t = (\mu - q) S_t \, dt + \sigma S_t \, dW_t
$$

By Itô's Lemma, the instantaneous 1-period logarithmic return over a small interval $\Delta t$ is strictly normally distributed (Gaussian):

$$
\ln\left(\frac{S_{t+\Delta t}}{S_t}\right) \sim \mathcal{N}\left(\left(\mu - q - \frac{1}{2}\sigma^2\right)\Delta t, \, \sigma^2 \Delta t\right)
$$

When compounded over the continuous horizon $T$, the cumulative log return remains normally distributed:

$$
\ln\left(\frac{S_T}{S_0}\right) \sim \mathcal{N}\left(\left(\mu - q - \frac{1}{2}\sigma^2\right)T, \, \sigma^2 T\right)
$$

Because these continuously compounded normal returns exponentiate over time, the terminal spot price $S_T$ is strictly distributed lognormally:

$$
S_T \sim \text{Lognormal}\left(\ln S_0 + \mu_{adj} T, \, \sigma^2 T\right)
$$

where $\mu_{adj} = \mu - q - \frac{1}{2}\sigma^2$.

**Model Parameters & Calibration**

* **Constant Volatility (**$\sigma$**):** The annualized sample standard deviation of historical daily log returns.
* **Physical Drift (**$\mu$**):** The real-world expected return utilizing the Capital Asset Pricing Model (CAPM): $\mu = r + \beta(\text{ERP}) + \alpha_{\text{drift}}/T$.
* **Risk-Free Rate (**$r$**) & Dividend Yield (**$q$**):** Continuous annual rates.

**Valuation & Simulation Process**
The terminal intrinsic value for an arbitrary set of multi-leg options is evaluated analytically as a discounted physical expectation:

$$
V_0 = e^{-rT} \sum_{j=1}^M w_j \left[ \phi_j S_0 e^{(\mu - q)T} \mathcal{N}(\phi_j d_{1,j}) - \phi_j K_j \mathcal{N}(\phi_j d_{2,j}) \right]
$$

where:

$$
d_{1,j} = \frac{\ln(S_0 / K_j) + \left(\mu - q + \frac{1}{2}\sigma^2\right)T}{\sigma \sqrt{T}}, \quad d_{2,j} = d_{1,j} - \sigma \sqrt{T}
$$

**Final Valuation Formula Variables:**

* $V_0$: The aggregate present expected value (premium) of the entire multi-leg option structure at time zero.
* $r$: The annualized continuous risk-free interest rate, used to discount the terminal expected value back to today.
* $T$: The time to expiration, expressed in years.
* $M$: The total number of unique option legs in the strategy (e.g., $M=4$ for an Iron Condor).
* $j$: The index tracking each specific leg in the summation ($1$ through $M$).
* $w_j$: The quantity multiplier or weight of leg $j$ (e.g., $+1$ for a standard long leg, $-2$ for the short body of a butterfly).
* $\phi_j$: The directional indicator for leg $j$, defined as $+1$ for Call options and $-1$ for Put options.
* $S_0$: The current spot price of the underlying asset.
* $\mu$: The physical expected drift of the asset, replacing the risk-neutral interest rate in the forward price projection.
* $q$: The continuous annualized dividend yield of the asset.
* $K_j$: The predetermined strike price associated with leg $j$.
* $\mathcal{N}(\cdot)$: The cumulative distribution function (CDF) of the standard normal distribution.
* $d_{1,j}$: The standardized normal variable representing the probability-weighted moneyness of the asset price projection for leg $j$.
* $d_{2,j}$: The standardized normal variable representing the absolute probability that leg $j$ will expire in-the-money.
* $\sigma$: The annualized constant standard deviation (volatility) of the asset's log returns.

---

### 2. Black-Scholes Mixture Model (Bootstrapped Empirical Volatility)

**Intuitive Foundation**
Volatility in real markets is non-stationary and regime-dependent. Rather than forcing a single historical average volatility, this model dynamically samples from historically realized rolling volatility windows. Because option payoffs are strictly convex with respect to volatility (positive Vomma), Jensen’s Inequality guarantees that sampling across a variance distribution yields a higher, more realistic option premium than evaluating at a single mean volatility.

**Distributional Assumptions**
As with the standard framework, the 1-period logarithmic returns over an interval $\Delta t$ are assumed to be normally distributed:

$$
\left.\ln\left(\frac{S_{t+\Delta t}}{S_t}\right) \right\vert{} \sigma_i \sim \mathcal{N}\left(\left(\mu - q - \frac{1}{2}\sigma_i^2\right)\Delta t, \, \sigma_i^2 \Delta t\right)
$$

Conditioned on a specific realized volatility state $\sigma_i$, compounding these normal returns over $T$ periods means the cumulative log return is Gaussian:

$$
\left.\ln\left(\frac{S_T}{S_0}\right) \right\vert{} \sigma_i \sim \mathcal{N}\left(\left(\mu - q - \frac{1}{2}\sigma_i^2\right)T, \, \sigma_i^2 T\right)
$$

Consequently, for any given volatility path, the terminal spot price $S_T$ remains lognormally distributed. Unconditionally, the final distribution of terminal spot prices is a continuous scale mixture of these lognormal distributions integrated across the empirical cumulative distribution function $\hat{F}_\sigma$:

$$
F(S_T) = \int_{0}^{\infty} \Phi\left(\frac{\ln(S_T/S_0) - \mu_{adj}(\sigma)T}{\sigma \sqrt{T}}\right) d\hat{F}_{\sigma}(\sigma)
$$

**Model Parameters & Calibration**

* **Empirical Rolling Volatility Set (**$\Omega_\sigma$**):** A discrete array of overlapping historical annualized standard deviations matching the trade duration:

$$
\Omega_\sigma = \{\sigma_D, \sigma_{D+1}, \dots, \sigma_N\}
$$

**Valuation & Simulation Process**
For each Monte Carlo path $i$, a historical volatility parameter $\sigma_i$ is drawn uniformly with replacement from $\Omega_\sigma$. The structure is priced analytically using that specific $\sigma_i$, and the final expected value is the arithmetic mean of all path valuations:

$$
V_0 = e^{-rT} \frac{1}{N_{paths}} \sum_{i=1}^{N_{paths}} \sum_{j=1}^M w_j \left[ \phi_j S_0 e^{(\mu - q)T} \mathcal{N}(\phi_j d_{1,j}^{(i)}) - \phi_j K_j \mathcal{N}(\phi_j d_{2,j}^{(i)}) \right]
$$

**Final Valuation Formula Variables:**

* $V_0$: The aggregate present expected value of the option structure.
* $e^{-rT}$: The continuous discount factor bringing the terminal value to present day using the risk-free rate $r$ and time $T$.
* $N_{paths}$: The total number of independent Monte Carlo simulation iterations.
* $i$: The index tracking the current simulation path ($1$ through $N_{paths}$).
* $M$: The total number of option legs in the structure.
* $j$: The index tracking each specific leg in the structure ($1$ through $M$).
* $w_j$: The positional quantity or weight of leg $j$.
* $\phi_j$: The Call ($+1$) or Put ($-1$) identifier for leg $j$.
* $S_0$: The current spot price of the asset.
* $\mu, q$: The expected physical drift and continuous dividend yield.
* $K_j$: The strike price of leg $j$.
* $\mathcal{N}(\cdot)$: The standard normal cumulative distribution function.
* $d_{1,j}^{(i)}$ and $d_{2,j}^{(i)}$: The analytical Black-Scholes probability inputs, dynamically calculated for path $i$ using the unique historical volatility parameter $\sigma_i$ drawn specifically for that path.

---

### 3. Black-Scholes Hybrid Mixture (Empirical Body + Stochastic EVT Volatility Tail)

**Intuitive Foundation**
The empirical bootstrap cannot simulate volatility higher than the historical sample maximum. To synthesize unprecedented market panics, this model splices the empirical volatility body with Extreme Value Theory (EVT). To account for parameter uncertainty, it utilizes **Method A (Asymptotic MLE Normality / Fisher Information Matrix)**. By extracting the Fisher Information Matrix (the inverse Hessian), Method A analytically maps the parameter uncertainty into a multivariate normal distribution, ensuring we do not underestimate the probability of extreme right-tail shape parameters without requiring heavy bootstrapping operations.

**Distributional Assumptions**
Because this is fundamentally a Black-Scholes framework, the underlying 1-period logarithmic returns over an interval $\Delta t$ are structurally assumed to be normally distributed:

$$
\left.\ln\left(\frac{S_{t+\Delta t}}{S_t}\right) \right\vert{} \sigma_i \sim \mathcal{N}\left(\left(\mu - q - \frac{1}{2}\sigma_i^2\right)\Delta t, \, \sigma_i^2 \Delta t\right)
$$

Compounded over $T$ periods, this naturally results in a lognormal distribution for the terminal stock price $S_T$ conditional on a given volatility.

However, the volatility parameter driving that lognormal distribution is stochastic. Below the 90th percentile threshold $u$, volatility follows the empirical distribution. Above $u$, the exceedances $y = \sigma - u$ follow a Generalized Pareto Distribution (GPD):

$$
G_{\xi, \beta}(y) = 1 - \left(1 + \frac{\xi y}{\beta}\right)^{-1/\xi}
$$

The composite cumulative distribution function for volatility is:

$$
F_{\text{Hybrid}}(\sigma) = \begin{cases} \hat{F}_n(\sigma) & \text{for } \sigma \le u \\\\ 0.90 + 0.10 \cdot G_{\xi, \beta}(\sigma - u) & \text{for } \sigma > u \end{cases}
$$

**Model Parameters & Stochastic Calibration (Method A)**

* **Threshold (**$u$**):** The 90th percentile of historical rolling volatilities.
* **Maximum Likelihood Estimates (**$\hat{\theta}$**):** The point estimates for shape ($\hat{\xi}$) and scale ($\hat{\beta}$) obtained by minimizing the negative log-likelihood of the exceedances.
* **Parameter Covariance (**$\Sigma$**):** The inverse Hessian matrix derived from the MLE optimization, representing the joint parameter uncertainty.

**Valuation & Simulation Process**
For each simulation path $i$:

1. Draw a unique, stochastic parameter pair $\theta_i$ from the multivariate normal parameter distribution:

$$
\begin{bmatrix} \xi_i \\\\ \beta_i \end{bmatrix} \sim \mathcal{N}\left( \begin{bmatrix} \hat{\xi} \\\\ \hat{\beta} \end{bmatrix}, \, \Sigma \right)
$$

2. Draw a uniform random variate $U^{(i)} \sim \mathcal{U}(0, 1)$.

3. Apply Inverse Transform Sampling to determine the path's volatility:

$$
\sigma_i = \begin{cases} \text{Quantile of } \hat{F}_n \text{ at } U^{(i)} & \text{if } U^{(i)} \le 0.90 \\\\ u + \frac{\beta_i}{\xi_i}\left[\left(\frac{1 - U^{(i)}}{0.10}\right)^{-\xi_i} - 1\right] & \text{if } U^{(i)} > 0.90 \end{cases}
$$

4. The structure is priced analytically using the drawn $\sigma_i$. The final expectation integrates Jensen's Inequality across both volatility regimes and parameter stochasticity:

$$
V_0 = e^{-rT} \frac{1}{N_{paths}} \sum_{i=1}^{N_{paths}} \sum_{j=1}^M w_j \left[ \phi_j S_0 e^{(\mu - q)T} \mathcal{N}(\phi_j d_{1,j}^{(i)}) - \phi_j K_j \mathcal{N}(\phi_j d_{2,j}^{(i)}) \right]
$$

**Final Valuation Formula Variables:**

* $V_0$: The aggregate present premium of the multi-leg option structure.
* $e^{-rT}$: The continuous risk-free discount factor.
* $N_{paths}$: The number of Monte Carlo paths generated (e.g., 10,000).
* $i$: The path index ($1$ to $N_{paths}$).
* $M$: The total number of legs in the evaluated structure.
* $j$: The leg index ($1$ to $M$).
* $w_j$: The weighting quantity of leg $j$.
* $\phi_j$: $+1$ for Calls, $-1$ for Puts.
* $S_0$: Current underlying asset spot price.
* $\mu, q$: Physical drift rate and dividend yield.
* $K_j$: The designated strike price of leg $j$.
* $\mathcal{N}(\cdot)$: The standard normal CDF.
* $d_{1,j}^{(i)}$ and $d_{2,j}^{(i)}$: The probability inputs computed using $\sigma_i$, where $\sigma_i$ is a single volatility draw sourced either from the empirical body or synthesized dynamically using the stochastic Generalized Pareto Distribution parameters ($\xi_i, \beta_i$) drawn for path $i$.

---

### 4. Return-Based EVT Model (Empirical Center + Stochastic EVT Tails)

**Intuitive Foundation**
This model simulates price trajectories step-by-step over daily intervals, entirely bypassing Gaussian integration. Normal days are sampled from the empirical center, while crashes and squeezes are drawn from GPD distributions. **Method A (Asymptotic MLE Normality)** is applied to the left and right tail parameters. Because option value is massively convex to the tail shape parameter $\xi$, integrating over the wide covariance matrix of the parameter fits significantly inflates the value of deep out-of-the-money options via Jensen's Inequality.

**Distributional Assumptions**
The daily logarithmic return $R_t = \ln(S_t / S_{t-1})$ follows a spliced three-regime distribution:

$$
F_R(r) = \begin{cases} \tau_L \left[1 + \frac{\xi_L (\vert{}r\vert{} - \vert{}u_L\vert{})}{\beta_L}\right]^{-1/\xi_L} & \text{for } r < u_L \\\\ \hat{F}_{\text{emp}}(r) & \text{for } u_L \le r \le u_R \\\\ 1 - \tau_R \left[1 + \frac{\xi_R (r - u_R)}{\beta_R}\right]^{-1/\xi_R} & \text{for } r > u_R \end{cases}
$$

where $\tau_L = \tau_R = 0.05$.

**Model Parameters & Stochastic Calibration (Method A)**

* **Thresholds (**$u_L, u_R$**):** The 5th and 95th percentiles of historical daily returns.
* **Left/Right Point Estimates:** $(\hat{\xi}_L, \hat{\beta}_L)$ and $(\hat{\xi}_R, \hat{\beta}_R)$.
* **Parameter Covariance (**$\Sigma_L, \Sigma_R$**):** The inverse Hessian matrices evaluated at the respective MLE peaks, mapping the boundary of our parameter ignorance.

**Valuation & Simulation Process**
For each simulation path $i$:

1. Draw stochastic left and right parameters from their respective multivariate normal distributions: $\theta_{L,i} \sim \mathcal{N}(\hat{\theta}_L, \Sigma_L)$ and $\theta_{R,i} \sim \mathcal{N}(\hat{\theta}_R, \Sigma_R)$.

2. For each trading day $t$ in $N_{days}$, draw a uniform variate $U_t \sim \mathcal{U}(0, 1)$:

   * If $U_t < 0.05$: $R_t = u_L - \text{GPD}^{-1}\left(\frac{U_t}{0.05}; \xi_{L,i}, \beta_{L,i}\right) - \bar{R}$.

   * If $U_t > 0.95$: $R_t = u_R + \text{GPD}^{-1}\left(\frac{U_t - 0.95}{0.05}; \xi_{R,i}, \beta_{R,i}\right) - \bar{R}$.

   * Otherwise: Sample randomly from the centered empirical body.

3. Compound the terminal spot price incorporating the physical drift correction:

$$
S_T^{(i)} = S_0 \exp\left( \sum_{t=1}^{N_{days}} R_t + \left(\mu - q - \frac{1}{2}\sigma_{\text{emp}}^2\right)T \right)
$$

4. The final option premium is the discounted arithmetic mean of the intrinsic payoffs across all multi-leg structures:

$$
V_0 = e^{-rT} \frac{1}{N_{paths}} \sum_{i=1}^{N_{paths}} \sum_{j=1}^M w_j \max(\phi_j (S_T^{(i)} - K_j), 0)
$$

**Final Valuation Formula Variables:**

* $V_0$: The aggregate expected present value of the structure.
* $e^{-rT}$: The standard continuous risk-free discount factor.
* $N_{paths}$: The total number of daily-stepped Monte Carlo price trajectories generated.
* $i$: The specific simulation path index ($1$ to $N_{paths}$).
* $M$: The total number of distinct option legs making up the trade.
* $j$: The leg index indicator ($1$ to $M$).
* $w_j$: The weighting multiplier (position size and side) for leg $j$.
* $\phi_j$: Directional flag ($+1$ for a Call, $-1$ for a Put).
* $S_T^{(i)}$: The simulated terminal spot price at expiration for path $i$, produced by compounding the daily returns $R_t$ (drawn from the stochastic GPD tails or empirical center) over $N_{days}$ and adjusted by the physical drift $\mu$.
* $K_j$: The strike price of leg $j$.
* $\max(\dots, 0)$: The standard intrinsic payoff function. It computes the cash value of the option at expiration $\phi_j(S_T^{(i)} - K_j)$, forcing the value to strictly zero if the option expires out-of-the-money.

---

### 5. Declustered Return-Based EVT Model (Runs Method)

**Intuitive Foundation**
Extreme market shocks cluster in time, violating the statistical independence assumption strictly required to fit an EVT distribution. A 30% drop over four days should be structurally viewed as one core panic event, not four separate independent Black Swans. The Runs Method addresses this by isolating true, independent block maxima.

Because this declustering shrinks the effective sample size of tail events (sometimes down to just 15 or 20 isolated historical crashes), epistemic parameter uncertainty becomes massive. **Method A (Fisher Information Matrix)** mathematically penalizes this sample scarcity: the tiny sample creates a wide, flat inverse Hessian matrix, resulting in massive variance draws for the shape parameters and properly forcing the model to heavily overprice the wings to compensate for structural blindness.

**Distributional Assumptions**
Let $I_t = \mathbb{I}_{\{R_t < u_L\}}$ be an indicator sequence for raw daily exceedances. An exceedance cluster $C_m$ is defined such that the time gap between exceedance days satisfies $t_{j+1} - t_j \le k$. The true independent variable representing the tail is the cluster maximum:

$$
\tilde{Y}_m = \max_{t \in C_m} (\vert{}R_t\vert{} - \vert{}u_{threshold}\vert{})
$$

The daily logarithmic return $R_t$ follows a spliced three-regime distribution, where the tails are calibrated strictly on these independent block maxima rather than raw, overlapping exceedances:

$$
F_R(r) = \begin{cases} \tau_L \left[1 + \frac{\xi_{L, dec} (\vert{}r\vert{} - \vert{}u_L\vert{})}{\beta_{L, dec}}\right]^{-1/\xi_{L, dec}} & \text{for } r < u_L \\\\ \hat{F}_{\text{emp}}(r) & \text{for } u_L \le r \le u_R \\\\ 1 - \tau_R \left[1 + \frac{\xi_{R, dec} (r - u_R)}{\beta_{R, dec}}\right]^{-1/\xi_{R, dec}} & \text{for } r > u_R \end{cases}
$$

**Model Parameters & Stochastic Calibration (Method A)**

* **Runs Window (**$k$**):** Set to 5 trading days to merge dependent consecutive shocks.
* **Declustered Covariance Matrices (**$\Sigma_{L, dec}, \Sigma_{R, dec}$**):** The inverse Hessians extracted from the MLE fits of the strictly independent peak vectors.

**Valuation & Simulation Process**
For each simulation path $i$:

1. Draw stochastic left and right parameters from their wide, declustered normal distributions: $\theta_{L, i} \sim \mathcal{N}(\hat{\theta}_{L, dec}, \Sigma_{L, dec})$ and $\theta_{R, i} \sim \mathcal{N}(\hat{\theta}_{R, dec}, \Sigma_{R, dec})$.

2. For each trading day $t$ in $N_{days}$, draw a uniform variate $U_t \sim \mathcal{U}(0, 1)$:

   * If $U_t < 0.05$: A standalone, independent crash is drawn using the inverse CDF of the declustered Left GPD: $R_t = u_L - \text{GPD}^{-1}\left(\frac{U_t}{0.05}; \xi_{L,i}, \beta_{L,i}\right) - \bar{R}$.

   * If $U_t > 0.95$: A standalone squeeze is drawn using the inverse CDF of the declustered Right GPD: $R_t = u_R + \text{GPD}^{-1}\left(\frac{U_t - 0.95}{0.05}; \xi_{R,i}, \beta_{R,i}\right) - \bar{R}$.

   * Otherwise: The daily return is sampled randomly from the centered empirical body.

3. Accumulate the daily returns, apply the continuous physical drift, and compute terminal spot price:

$$
S_T^{(i)} = S_0 \exp\left( \sum_{t=1}^{N_{days}} R_t + \left(\mu - q - \frac{1}{2}\sigma_{\text{emp}}^2\right)T \right)
$$

4. The final option premium is the discounted arithmetic mean of the intrinsic payoffs:

$$
V_0 = e^{-rT} \frac{1}{N_{paths}} \sum_{i=1}^{N_{paths}} \sum_{j=1}^M w_j \max(\phi_j (S_T^{(i)} - K_j), 0)
$$

**Final Valuation Formula Variables:**

* $V_0$: The aggregate calculated premium of the multi-leg structure today.
* $e^{-rT}$: The continuous interest-rate discount mechanism.
* $N_{paths}$: The total sequence of independent daily-compounded simulations.
* $i$: The current iteration index for the path array ($1$ to $N_{paths}$).
* $M$: The total number of legs in the custom option strategy.
* $j$: The leg identifier ($1$ to $M$).
* $w_j$: The directional weighting (quantity) of leg $j$.
* $\phi_j$: The Call/Put identifier multiplier ($+1$ or $-1$).
* $S_T^{(i)}$: The final price of the underlying asset at expiration for path $i$, produced by compounding daily returns $R_t$. Crucially, the extreme tail jumps in this path are dictated by the declustered parameter vectors ($\theta_{L, i}, \theta_{R, i}$), ensuring the simulation does not over-cluster shocks while accurately compensating for sample scarcity.
* $K_j$: The required strike price for leg $j$.
* $\max(\dots, 0)$: The mathematical floor ensuring the terminal option payout never drops below zero cash value.

---

### 6. GARCH-EVT Filtered Historical Simulation ($L_2$ Norm)

**Intuitive Foundation**
Treats volatility as a path-dependent autoregressive process where shocks instantly spike future baseline volatility. Because Parametric Bootstrapping destroys the sequential time-dependency of autoregressive models (unless using computationally prohibitive Filtered Residual Bootstrapping), this model utilizes **Method A (Asymptotic MLE Normality / Fisher Information Matrix)**. Method A mathematically extracts the joint uncertainty directly from the curvature of the time-dependent log-likelihood function.

**Distributional Assumptions**
Returns are decomposed into a conditional mean, conditional standard deviation $\sigma_t$, and a standardized innovation $Z_t$:

$$
R_t = \mu_t + \sigma_t Z_t
$$

The conditional variance $\sigma_t^2$ updates via an $L_2$ norm autoregressive process:

$$
\sigma_{t+1}^2 = \omega + \alpha (R_t - \mu_t)^2 + \beta_{garch} \sigma_t^2
$$

The standardized innovation $Z_t$ follows the three-regime spliced EVT distribution, calibrated to the residuals $z_t = \frac{R_t - \bar{R}}{\sigma_t}$.

**Model Parameters & Stochastic Calibration (Method A)**

* **GARCH Point Estimates:** $\hat{\theta}_{garch} = [\hat{\omega}, \hat{\alpha}, \hat{\beta}_{garch}]^T$.
* **GARCH Covariance (**$\Sigma_{garch}$**):** The $3 \times 3$ inverse Hessian matrix derived from the GARCH Gaussian log-likelihood optimization.
* **Residual Tail Covariance (**$\Sigma_Z$**):** The inverse Hessian from the EVT fit of the standardized residuals.

**Valuation & Simulation Process**
For each simulation path $i$:

1. Draw stochastic GARCH parameters from the multivariate normal likelihood distribution:

$$
\theta_{garch, i} \sim \mathcal{N}(\hat{\theta}_{garch}, \Sigma_{garch})
$$

2. Draw stochastic tail parameters for the residuals: $\theta_{Z, i} \sim \mathcal{N}(\hat{\theta}_Z, \Sigma_Z)$.

3. For each day $t$, draw a standardized shock $Z_t$ from the stochastic EVT-residual distribution.

4. Calculate the raw return $r_t = \mu_t + \sigma_t Z_t$ and advance the spot price.

5. Recursively update the conditional variance for the next day using the path's unique, stochastic GARCH parameters:

$$
\sigma_{t+1}^2 = \omega_i + \alpha_i (\sigma_t Z_t)^2 + \beta_{garch, i} \sigma_t^2
$$

6. Determine the path's terminal spot price:

$$
S_T^{(i)} = S_0 \exp\left( \sum_{t=1}^{N_{days}} r_t \right)
$$

7. Calculate the final discounted expected value across all simulated terminal payouts:

$$
V_0 = e^{-rT} \frac{1}{N_{paths}} \sum_{i=1}^{N_{paths}} \sum_{j=1}^M w_j \max(\phi_j (S_T^{(i)} - K_j), 0)
$$

**Final Valuation Formula Variables:**

* $V_0$: The final calculated premium of the option structure.
* $e^{-rT}$: The continuous time-value discount factor.
* $N_{paths}$: The total count of sequential Monte Carlo simulations executed.
* $i$: The iteration tracker for each distinct path ($1$ to $N_{paths}$).
* $M$: The number of independent legs evaluated inside the structure.
* $j$: The specific leg index ($1$ to $M$).
* $w_j$: The signed quantity weight of leg $j$.
* $\phi_j$: Call ($+1$) or Put ($-1$) orientation multiplier.
* $S_T^{(i)}$: The simulated expiration price of the asset for path $i$. Unlike earlier models, this price is the result of a highly path-dependent process. The simulated daily returns $r_t$ dynamically alter the variance $\sigma_{t+1}^2$ via the path's unique stochastic GARCH parameters ($\omega_i, \alpha_i, \beta_i$), directly accelerating or dampening future volatility regimes based strictly on the magnitude of the standardized shock $Z_t$.
* $K_j$: The strike price of leg $j$.
* $\max(\dots, 0)$: The intrinsic payoff floor ensuring positive or zero terminal cash payouts.

---

### 7. MAD-Based Filtered Historical Simulation ($L_1$ Norm + EVT)

**Intuitive Foundation**
Because fat-tailed financial returns possess infinite theoretical fourth moments ($\mathbb{E}[R^4] = \infty$), the $L_2$ squared-error metric in standard GARCH causes the variance filter to destabilize and explode during severe historical crashes. This model replaces the fragile $L_2$ filter with a robust $L_1$ Mean Absolute Deviation (MAD) filter. Like Model 6, it relies on **Method A (Fisher Information Matrix)** to extract parameter uncertainty for the residuals without breaking sequential time structures.

**Distributional Assumptions**
The daily price process follows:

$$
R_t = \mu_t + \text{MAD}_t \cdot Z_t
$$

The $L_1$ local scale parameter $\text{MAD}_t = \mathbb{E}[\vert{}R_t - \mathbb{E}[R_t]\vert{}]$ updates via an Exponentially Weighted Moving Average (EWMA):

$$
\text{MAD}_{t+1} = \lambda \text{MAD}_t + (1 - \lambda)\vert{}R_t - \mu_t\vert{}
$$

Standardized $L_1$ residuals $z_t = \frac{R_t}{\text{MAD}_t}$ are declustered via the Runs Method, and their tails are modeled via a GPD.

**Model Parameters & Stochastic Calibration (Method A)**

* **Decay Factor (**$\lambda$**):** Statically fixed at $0.94$ (RiskMetrics baseline) to ensure filter stability.
* **Initial Scale (**$\text{MAD}_1$**):** Empirical sample mean of historical absolute returns.
* $L_1$ **Tail Covariance (**$\Sigma_{Z, dec}$**):** The inverse Hessian extracted from the MLE fit of the declustered $L_1$-normalized residual exceedances.

**Valuation & Simulation Process**
For each simulation path $i$:

1. Draw stochastic parameters for the left and right residual tails: $\theta_{Z, i} \sim \mathcal{N}(\hat{\theta}_{Z, dec}, \Sigma_{Z, dec})$.

2. Initialize scale at the latest market state: $\text{MAD}_1 = \text{MAD}_{\text{latest}}$.

3. For each trading day $t$:

   * Draw a residual shock $z_t$ from the stochastic declustered spliced distribution.

   * Calculate the raw return jump: $r_{\text{sim}} = z_t \cdot \text{MAD}_t$.

   * Compute physical drift using the annualized $L_1$ scale adjustment:

$$
\mu_t = \left(\mu - q - \frac{1}{2}(\text{MAD}_t^2 \cdot 252)\right) \Delta t
$$

* Advance spot price: $S_{t+1} = S_t \exp(r_{\text{sim}} + \mu_t)$.

* Recursively update the $L_1$ scale parameter for the next day:

$$
\text{MAD}_{t+1} = \lambda \text{MAD}_t + (1 - \lambda)\vert{}r_{\text{sim}}\vert{}
$$

4. Calculate terminal intrinsic payoffs and discount to present value.

**Final Valuation Formula Variables:**

* $V_0$: The overall expected present value of the chosen option matrix.
* $e^{-rT}$: The continuous-time discount factor mapping expiration values to present day.
* $N_{paths}$: The total count of $L_1$-filtered simulated price trajectories.
* $i$: The path tracking iteration identifier ($1$ to $N_{paths}$).
* $M$: The aggregate count of option legs evaluated.
* $j$: The individual leg index identifier ($1$ to $M$).
* $w_j$: The weighting ratio or sizing multiplier representing leg $j$.
* $\phi_j$: The positive ($+1$) or negative ($-1$) flag determining Call or Put intrinsic payoff logic.
* $S_T^{(i)}$: The simulated spot price at expiration for path $i$. In this robust model, $S_T$ is constructed using sequential daily jump dynamics $r_{\text{sim}, t}$ generated from standardized $L_1$ residuals $z_t$ drawn from the stochastically parametrized GPD. The core innovation is that the volatility metric scaling these residuals, $\text{MAD}_t$, is immune to explosive behavior during $L_2$ tail events.
* $K_j$: The designated strike target for leg $j$.
* $\max(\dots, 0)$: The standard non-linear payoff ceiling applied at expiration, prohibiting negative values.

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
  
$$
\Delta = \frac{\partial V}{\partial S} = \sum_{j=1}^M w_j \phi_j e^{-q T} \mathcal{N}(\phi_j d_{1,j})
$$

* **Vega ($\nu$):** Sensitivity to a 1% absolute move in implied volatility:
  
$$
\nu = \frac{1}{100} \frac{\partial V}{\partial \sigma} = \frac{1}{100} \sum_{j=1}^M w_j S_0 e^{-q T} \phi_j n(d_{1,j}) \sqrt{T}
$$

* **Theta ($\Theta$):** Calendar day time decay:
  
$$
\Theta_{\text{call}} = \frac{1}{365}\left[ -\frac{S_0 n(d_1)\sigma e^{-q T}}{2\sqrt{T}} - r K e^{-r T} \mathcal{N}(d_2) + q S_0 e^{-q T} \mathcal{N}(d_1) \right]
$$
  
$$
\Theta_{\text{put}} = \frac{1}{365}\left[ -\frac{S_0 n(d_1)\sigma e^{-q T}}{2\sqrt{T}} + r K e^{-r T} \mathcal{N}(-d_2) - q S_0 e^{-q T} \mathcal{N}(-d_1) \right]
$$

* **Rho ($\rho$):** Sensitivity to a 1% change in the risk-free interest rate:
  
$$
\rho = \frac{1}{100} \frac{\partial V}{\partial r} = \frac{1}{100} \sum_{j=1}^M w_j \phi_j K_j T e^{-r T} \mathcal{N}(\phi_j d_{2,j})
$$

### Second-Order & Cross Greeks
* **Gamma ($\Gamma$):** Curvature and directional acceleration:
  
$$
\Gamma = \frac{\partial^2 V}{\partial S^2} = \sum_{j=1}^M w_j \frac{e^{-q T} n(d_{1,j})}{S_0 \sigma \sqrt{T}}
$$

* **Vomma (Volga):** Volatility convexity (rate of change of Vega with respect to volatility):
  
$$
\text{Vomma} = \frac{\partial^2 V}{\partial \sigma^2} = \nu \cdot \frac{d_{1} d_{2}}{\sigma}
$$

* **Vanna:** Cross-derivative of option value with respect to spot price and volatility:
  
$$
\text{Vanna} = \frac{\partial^2 V}{\partial S \partial \sigma} = -e^{-q T} n(d_1) \frac{d_2}{\sigma}
$$

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10+
* Git

### Local Deployment
1. Clone the repository:
   
~~~bash
   git clone https://github.com/yourusername/option-valuation-tool.git
   cd option-valuation-tool
~~~

2. Create and activate a virtual environment:
   
~~~bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
~~~

3. Install required dependencies:
   
~~~bash
   pip install -r requirements.txt
~~~

4. Launch the Streamlit dashboard:
   
~~~bash
   streamlit run app.py
~~~

---

## 📦 Dependencies
* **Streamlit:** Web interface architecture and interactive state management.
* **NumPy:** Vectorized Monte Carlo trajectory simulations and numerical linear algebra.
* **Pandas:** Time series processing and business holiday calendar math.
* **SciPy:** Maximum Likelihood Estimation optimization and Extreme Value distributions.
* **Plotly:** Interactive terminal payoff charts and 3D risk surfaces.
* **yfinance:** Market data extraction with underlying proxy mapping.