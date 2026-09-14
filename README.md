# \# Quantitative Option Valuation \& Fat-Tail Risk Analysis Tool

# 

# A high-performance quantitative finance and options engineering application built with Streamlit, Plotly, SciPy, and NumPy. 

# 

# This platform replaces the Gaussian and constant-volatility assumptions of classical financial engineering with non-linear volatility dynamics and Extreme Value Theory (EVT). It evaluates single-leg options and complex multi-leg combinations across physical real-world probability distributions $\\mathbb{P}$ rather than artificial risk-neutral surfaces $\\mathbb{Q}$, capturing volatility clustering, asymmetric tail decay, parameter estimation risk, and Jensen's inequality convexity.

# 

# \---

# 

# \## 📐 Theoretical Framework: Physical Valuation vs. Risk-Neutral Pricing

# 

# Standard option pricing models operate under the risk-neutral measure $\\mathbb{Q}$, discounting expected terminal payoffs at the risk-free rate $r$ under the assumption that all underlying assets drift at $r - q$. While mathematically convenient for dynamic delta-hedging in complete markets, $\\mathbb{Q}$-pricing fails to reflect:

# 1\. \*\*Physical Expectation $\\mathbb{P}$:\*\* Real-world assets compound with an equity risk premium (ERP) and idiosyncratic alpha: $\\mu = r + \\beta(\\text{ERP}) + \\alpha\_{\\text{drift}}/T$.

# 2\. \*\*Paretian Tail Power Laws:\*\* Equity crash distributions exhibit heavy tails characterized by Generalized Pareto Distributions where extreme outcomes decay as power laws $P(X > x) \\sim x^{-\\alpha}$, rather than exponentially as in Gaussian models.

# 3\. \*\*Volatility Convexity (Jensen's Inequality):\*\* Because option contracts possess positive Vega and Vomma ($\\frac{\\partial^2 V}{\\partial \\sigma^2} > 0$), integrating over an uncertain or fluctuating volatility distribution yields an expected value strictly greater than evaluating the contract at the mean volatility:

# &#x20;  $$\\mathbb{E}\[V(\\sigma)] \\ge V(\\mathbb{E}\[\\sigma])$$

# 

# \---

# 

# \## 🧠 Comprehensive Model Hierarchy \& Mathematical Formulations

# 

# The tool implements an evolutionary hierarchy of seven valuation engines. Each engine progressively strips away artificial constraints to model real-world asset price dynamics.

# 

# ```text

# \[1. Standard Black-Scholes] 

# &#x20;      │ (Relax constant volatility)

# &#x20;      ▼

# \[2. Empirical Bootstrap Mixture] 

# &#x20;      │ (Overcome historical max volatility truncation)

# &#x20;      ▼

# \[3. Hybrid Mixture (EVT Volatility Tail)] 

# &#x20;      │ (Transition from macro-vol mixture to daily return simulation)

# &#x20;      ▼

# \[4. Return-Based EVT (Empirical Body + GPD Tails)] 

# &#x20;      │ (Filter dependent extreme clusters via Runs Method)

# &#x20;      ▼

# \[5. Declustered EVT (Runs Method Block Maxima)] 

# &#x20;      │ (Introduce autoregressive conditional heteroskedasticity)

# &#x20;      ▼

# \[6. GARCH(1,1)-EVT Filtered Historical Simulation (L2 Norm)] 

# &#x20;      │ (Overcome infinite fourth-moment instability of squared returns)

# &#x20;      ▼

# \[7. MAD-Based Filtered Historical Simulation (L1 Norm + EVT)]

# 

# \---

# 

# \### Model 1: Standard Black-Scholes (Analytical Constant Volatility)

# 

# \#### 1. Conceptual Intuition

# Values the option structure analytically under the classical Black-Scholes-Merton assumption that the underlying asset follows continuous Geometric Brownian Motion (GBM) with constant volatility $\\sigma$ and constant drift. It ignores fat tails, skew, and market shocks.

# 

# \#### 2. Probabilistic Notation \& Distributional Assumptions

# Under the physical probability measure $\\mathbb{P}$, the continuous price process $S\_t$ satisfies the Stochastic Differential Equation (SDE):

# $$dS\_t = (\\mu - q) S\_t \\, dt + \\sigma S\_t \\, dW\_t$$

# where $W\_t$ is a standard Brownian motion under $\\mathbb{P}$, $\\mu$ is the subjective physical drift, and $q$ is the continuous dividend yield.

# 

# By Itô's Lemma, the log return over horizon $T$ follows a Gaussian distribution:

# $$\\ln\\left(\\frac{S\_T}{S\_0}\\right) \\sim \\mathcal{N}\\left(\\left(\\mu - q - \\frac{1}{2}\\sigma^2\\right)T, \\, \\sigma^2 T\\right)$$

# The terminal price $S\_T$ is distributed lognormally:

# $$S\_T \\sim \\text{Lognormal}\\left(\\ln S\_0 + \\mu\_{adj} T, \\, \\sigma^2 T\\right), \\quad \\text{where } \\mu\_{adj} = \\mu - q - \\frac{1}{2}\\sigma^2$$

# 

# \#### 3. Calibration \& Parameter Estimation

# \* \*\*Volatility ($\\sigma$):\*\* Sample standard deviation of historical daily log returns scaled to an annual horizon:

# &#x20; $$\\sigma = \\sqrt{\\frac{252}{N-1} \\sum\_{k=1}^N (R\_k - \\bar{R})^2}$$

# \* \*\*Physical Drift ($\\mu$):\*\* User-calibrated Capital Asset Pricing Model (CAPM) structure:

# &#x20; $$\\mu = r + \\beta \\cdot \\text{ERP} + \\frac{\\alpha\_{\\text{drift}}}{T}$$

# 

# \#### 4. Valuation \& Simulation Process

# The terminal intrinsic value for an arbitrary set of call/put legs $j \\in \\{1, \\dots, M\\}$ with strike $K\_j$, direction indicator $\\phi\_j \\in \\{+1 \\text{ (call)}, -1 \\text{ (put)}\\}$, and position weight $w\_j$ is evaluated analytically:

# $$V\_0 = e^{-rT} \\sum\_{j=1}^M w\_j \\left\[ \\phi\_j S\_0 e^{(\\mu - q)T} \\mathcal{N}(\\phi\_j d\_{1,j}) - \\phi\_j K\_j \\mathcal{N}(\\phi\_j d\_{2,j}) \\right]$$

# where:

# $$d\_{1,j} = \\frac{\\ln(S\_0 / K\_j) + \\left(\\mu - q + \\frac{1}{2}\\sigma^2\\right)T}{\\sigma \\sqrt{T}}, \\quad d\_{2,j} = d\_{1,j} - \\sigma \\sqrt{T}$$

# To map percentile distributions, terminal prices are simulated via:

# $$S\_T^{(i)} = S\_0 \\exp\\left(\\left(\\mu - q - \\frac{1}{2}\\sigma^2\\right)T + \\sigma \\sqrt{T} Z^{(i)}\\right), \\quad Z^{(i)} \\stackrel{\\text{i.i.d.}}{\\sim} \\mathcal{N}(0, 1)$$

# 

# \---

# 

# \### Model 2: Black-Scholes Mixture Model (Bootstrapped Empirical Volatility)

# 

# \#### 1. Conceptual Intuition

# Recognizes that volatility is not constant over time. Rather than assuming a single static $\\sigma$, this model repeatedly samples from historically realized rolling volatility regimes matching the duration of the trade ($N$ trading days to expiry). 

# 

# \#### 2. Probabilistic Notation \& Distributional Assumptions

# Conditioned on a specific volatility state $\\sigma\_i$, log returns are locally Gaussian:

# $$\\left.\\ln\\left(\\frac{S\_T}{S\_0}\\right) \\right| \\sigma\_i \\sim \\mathcal{N}\\left(\\left(\\mu - q - \\frac{1}{2}\\sigma\_i^2\\right)T, \\, \\sigma\_i^2 T\\right)$$

# The marginal distribution of terminal returns across the entire sample space constitutes a continuous scale mixture of Gaussians:

# $$F(S\_T) = \\int\_{0}^{\\infty} \\Phi\\left(\\frac{\\ln(S\_T/S\_0) - \\mu\_{adj}(\\sigma)T}{\\sigma \\sqrt{T}}\\right) dF\_{\\sigma}(\\sigma)$$

# where $F\_{\\sigma}$ is the empirical cumulative distribution function (ECDF) of rolling historical volatility.

# 

# \#### 3. Calibration \& Parameter Estimation

# For a trade with $D$ trading days to expiration:

# 1\. Compute the series of overlapping $D$-day historical annualized standard deviations across the full price history:

# &#x20;  $$\\sigma\_{\\tau} = \\sqrt{\\frac{252}{D-1} \\sum\_{k=\\tau-D+1}^{\\tau} (R\_k - \\bar{R}\_{\\tau})^2}, \\quad \\tau \\in \[D, N]$$

# 2\. Compile the discrete empirical set $\\Omega\_{\\sigma} = \\{\\sigma\_D, \\sigma\_{D+1}, \\dots, \\sigma\_N\\}$.

# 

# \#### 4. Valuation \& Simulation Process

# 1\. For each simulation path $i \\in \\{1, \\dots, \\text{num\\\_simulations}\\}$:

# &#x20;  \* Draw a volatility parameter uniformly with replacement: $\\sigma\_i \\sim \\text{Uniform}(\\Omega\_{\\sigma})$.

# &#x20;  \* Compute the analytical Black-Scholes structure value $V(\\sigma\_i)$ using sample volatility $\\sigma\_i$.

# &#x20;  \* Generate terminal spot price $S\_T^{(i)} = S\_0 \\exp\\left((\\mu - q - \\frac{1}{2}\\sigma\_i^2)T + \\sigma\_i \\sqrt{T} Z^{(i)}\\right)$, where $Z^{(i)} \\sim \\mathcal{N}(0,1)$.

# 2\. Calculate the discounted expected structural value:

# &#x20;  $$\\mathbb{E}\[V] = e^{-rT} \\frac{1}{M} \\sum\_{i=1}^M V(\\sigma\_i)$$

# 

# \---

# 

# \### Model 3: Black-Scholes Hybrid Mixture Model (Empirical Body + EVT Volatility Tail)

# 

# \#### 1. Conceptual Intuition

# Model 2 suffers from \*\*empirical truncation\*\*: it cannot simulate a volatility regime higher than the historical sample maximum. If an asset has never experienced a severe market dislocation within its specific data window, Model 2 understates extreme tail convexity. Model 3 splices the empirical volatility distribution with a Generalized Pareto Distribution (GPD) fitted to the top 10% extreme volatility exceedances.

# 

# \#### 2. Probabilistic Notation \& Distributional Assumptions

# Let $\\Sigma$ denote the rolling volatility random variable. The tail distribution above threshold $u$ (the 90th percentile $u = F\_{\\Sigma}^{-1}(0.90)$) is modeled via the Pickands-Balkema-de Haan Theorem:

# $$\\mathbb{P}(\\Sigma - u \\le y \\mid \\Sigma > u) \\approx G\_{\\xi, \\beta}(y) = 1 - \\left(1 + \\frac{\\xi y}{\\beta}\\right)^{-1/\\xi}$$

# where:

# \* $\\xi \\in \\mathbb{R}$ is the GPD shape parameter (tail index $\\alpha\_{\\text{tail}} = 1/\\xi$).

# \* $\\beta > 0$ is the scale parameter.

# \* Support is $y \\ge 0$ for $\\xi \\ge 0$, and $0 \\le y \\le -\\beta/\\xi$ for $\\xi < 0$.

# 

# The composite cumulative distribution function for volatility is:

# $$F\_{\\text{Hybrid}}(\\sigma) = \\begin{cases} 

# \\hat{F}\_n(\\sigma) \& \\text{for } \\sigma \\le u \\\\

# (1 - 0.10) + 0.10 \\cdot G\_{\\xi, \\beta}(\\sigma - u) \& \\text{for } \\sigma > u 

# \\end{cases}$$

# 

# \#### 3. Calibration \& Parameter Estimation

# 1\. Compute the empirical 90th percentile threshold: $u = \\text{Percentile}(\\Omega\_{\\sigma}, 90)$.

# 2\. Extract historical exceedances: $Y = \\{\\sigma - u \\mid \\sigma \\in \\Omega\_{\\sigma}, \\, \\sigma > u\\}$.

# 3\. Fit shape ($\\xi$) and scale ($\\beta$) via Maximum Likelihood Estimation (MLE):

# &#x20;  $$\\mathcal{L}(\\xi, \\beta; Y) = -k \\ln \\beta - \\left(1 + \\frac{1}{\\xi}\\right) \\sum\_{j=1}^k \\ln\\left(1 + \\frac{\\xi y\_j}{\\beta}\\right)$$

# 

# \#### 4. Valuation \& Simulation Process

# 1\. Draw standard uniform random variate $U^{(i)} \\sim \\mathcal{U}(0, 1)$.

# 2\. Apply Inverse Transform Sampling:

# &#x20;  $$\\sigma\_i = \\begin{cases} 

# &#x20;  \\text{quantile of } \\hat{F}\_n \\text{ at } U^{(i)} \& \\text{if } U^{(i)} \\le 0.90 \\\\

# &#x20;  u + \\frac{\\beta}{\\xi}\\left\[\\left(\\frac{1 - U^{(i)}}{0.10}\\right)^{-\\xi} - 1\\right] \& \\text{if } U^{(i)} > 0.90 

# &#x20;  \\end{cases}$$

# 3\. Evaluate analytical option values at $\\sigma\_i$ and simulate $S\_T^{(i)}$ paths using sampled $\\sigma\_i$.

# 

# \---

# 

# \### Model 4: Return-Based EVT Model (Empirical Center + EVT Tails)

# 

# \#### 1. Conceptual Intuition

# Abandons Black-Scholes analytical integration entirely. Simulates price paths step-by-step using a semi-parametric daily return distribution. Normal trading days are sampled directly from the empirical center, while the extreme left (crashes) and right (squeezes) tails are generated from parametric Generalized Pareto Distributions fitted to the historical 5th and 95th percentiles.

# 

# \#### 2. Probabilistic Notation \& Distributional Assumptions

# Let $R\_t = \\ln(S\_t / S\_{t-1})$ denote daily log returns. The distribution $F\_R(r)$ is partitioned into three regimes:

# $$F\_R(r) = \\begin{cases} 

# \\tau\_L \\left\[1 + \\frac{\\xi\_L (|r| - |u\_L|)}{\\beta\_L}\\right]^{-1/\\xi\_L} \& \\text{for } r < u\_L \\quad (\\text{Left Crash Tail}) \\\\

# \\hat{F}\_{\\text{emp}}(r) \& \\text{for } u\_L \\le r \\le u\_R \\quad (\\text{Empirical Body}) \\\\

# 1 - \\tau\_R \\left\[1 + \\frac{\\xi\_R (r - u\_R)}{\\beta\_R}\\right]^{-1/\\xi\_R} \& \\text{for } r > u\_R \\quad (\\text{Right Squeeze Tail})

# \\end{cases}$$

# where $u\_L$ is the 5th percentile return, $u\_R$ is the 95th percentile return, and $\\tau\_L = \\tau\_R = 0.05$.

# 

# \#### 3. Calibration \& Parameter Estimation

# 1\. Determine thresholds: $u\_L = \\text{Percentile}(R, 5)$ and $u\_R = \\text{Percentile}(R, 95)$.

# 2\. Form exceedance vectors:

# &#x20;  \* Left tail losses: $Y\_L = |r| - |u\_L|$ for all $r < u\_L$.

# &#x20;  \* Right tail gains: $Y\_R = r - u\_R$ for all $r > u\_R$.

# 3\. Maximize GPD log-likelihood to obtain $(\\xi\_L, \\beta\_L)$ and $(\\xi\_R, \\beta\_R)$.

# 4\. Extract centered interior returns: $R\_{\\text{center}} = \\{r - \\bar{R} \\mid u\_L \\le r \\le u\_R\\}$.

# 

# \#### 4. Valuation \& Simulation Process

# For each simulation path $i$:

# 1\. For each discrete trading day $t \\in \\{1, \\dots, D\\}$:

# &#x20;  \* Draw uniform variate $U\_t \\sim \\mathcal{U}(0, 1)$.

# &#x20;  \* If $U\_t < 0.05$, draw a left-tail loss: $R\_t = u\_L - \\text{GPD}^{-1}(U\_t/0.05; \\, \\xi\_L, \\beta\_L) - \\bar{R}$.

# &#x20;  \* If $U\_t > 0.95$, draw a right-tail gain: $R\_t = u\_R + \\text{GPD}^{-1}((U\_t - 0.95)/0.05; \\, \\xi\_R, \\beta\_R) - \\bar{R}$.

# &#x20;  \* Otherwise, sample randomly with replacement from $R\_{\\text{center}}$.

# 2\. Apply drift and compound terminal spot:

# &#x20;  $$S\_T^{(i)} = S\_0 \\exp\\left(\\sum\_{t=1}^D R\_t^{(i)} + \\left(\\mu - q - \\frac{1}{2}\\sigma\_{\\text{hist}}^2\\right) D \\cdot \\Delta t\\right), \\quad \\Delta t = \\frac{1}{252}$$

# 3\. Price structure: $\\mathbb{E}\[V] = e^{-rT} \\frac{1}{M} \\sum\_{i=1}^M \\text{Payoff}(S\_T^{(i)})$.

# 

# \---

# 

# \### Model 5: Declustered Return-Based EVT Model (Runs Method)

# 

# \#### 1. Conceptual Intuition

# Standard EVT requires returns to be independent and identically distributed (i.i.d.). In financial markets, large price shocks cluster together in time (volatility clustering). Fitting EVT directly to dependent raw returns violates the asymptotic assumptions of the Fisher-Tippett-Gnedenko theorem, overstating the frequency of independent black swans. 

# 

# Model 5 implements the \*\*Runs Method\*\* ($k=5$ day window) to cluster dependent exceedances, extracting only the single maximum shock per cluster to fit true, independent block maxima.

# 

# \#### 2. Probabilistic Notation \& Distributional Assumptions

# Let $\\{I\_t\\}\_{t=1}^N$ be an indicator sequence where $I\_t = \\mathbb{I}\_{\\{R\_t < u\_L\\}}$. An exceedance cluster $C\_m = \\{R\_{t\_1}, R\_{t\_2}, \\dots, R\_{t\_p}\\}$ is defined such that:

# $$t\_{j+1} - t\_j \\le k, \\quad \\forall j \\in \\{1, \\dots, p-1\\}$$

# where $k=5$ trading days defines the cluster run length. The cluster maximum loss is:

# $$\\tilde{Y}\_m = \\max\_{t \\in C\_m} (|R\_t| - |u\_L|)$$

# The resulting declustered exceedances $\\{\\tilde{Y}\_m\\}\_{m=1}^{N\_c}$ converge asymptotically to a generalized Pareto distribution under the Extremal Index $\\theta \\in (0, 1]$:

# $$\\mathbb{P}(\\tilde{Y} \\le y) = 1 - \\left(1 + \\frac{\\xi\_D y}{\\beta\_D}\\right)^{-1/\\xi\_D}$$

# 

# \#### 3. Calibration \& Parameter Estimation

# 1\. Scan historical returns $R\_t$ chronologically. Group consecutive exceedances separated by fewer than $k=5$ days into a unified cluster.

# 2\. Isolate the extreme peak within each cluster:

# &#x20;  \* Left tail: $\\tilde{y}\_{L, m} = |u\_L| - \\min\_{t \\in C\_m}(R\_t)$

# &#x20;  \* Right tail: $\\tilde{y}\_{R, m} = \\max\_{t \\in C\_m}(R\_t) - u\_R$

# 3\. Fit declustered GPD parameters $(\\xi\_{L, \\text{dec}}, \\beta\_{L, \\text{dec}})$ and $(\\xi\_{R, \\text{dec}}, \\beta\_{R, \\text{dec}})$ using MLE exclusively on the independent peak vector.

# 

# \#### 4. Valuation \& Simulation Process

# Path generation follows the daily step sampling protocol of Model 4, but draws tail innovations from the declustered GPD distributions $(\\xi\_{\\text{dec}}, \\beta\_{\\text{dec}})$. This prevents artificial amplification of tail risk caused by serially dependent crash runs.

# 

# \---

# 

# \### Model 6: GARCH-EVT Filtered Historical Simulation ($L\_2$ Norm)

# 

# \#### 1. Conceptual Intuition

# Combines parametric conditional heteroskedasticity with non-parametric EVT innovations. Standard EVT assumes stationary variance. In reality, a large market shock today directly increases conditional volatility tomorrow. 

# 

# Model 6 fits a GARCH(1,1) model to historical returns, standardizes the returns into independent identically distributed (i.i.d.) residuals, fits EVT to those standardized residuals, and simulates paths where daily shocks recursively feed back into next-day volatility.

# 

# \#### 2. Probabilistic Notation \& Distributional Assumptions

# The return process is decomposed into a conditional mean, conditional variance, and standardized innovation:

# $$R\_t = \\mu\_t + \\sigma\_t Z\_t, \\quad Z\_t \\stackrel{\\text{i.i.d.}}{\\sim} F\_Z(0, 1)$$

# The conditional variance $\\sigma\_t^2$ evolves according to a GARCH(1,1) process ($L\_2$ norm):

# $$\\sigma\_t^2 = \\omega + \\alpha \\epsilon\_{t-1}^2 + \\beta \\sigma\_{t-1}^2$$

# where $\\epsilon\_{t-1} = R\_{t-1} - \\bar{R}$ is the demeaned return shock. Stationarity requires $\\omega > 0, \\, \\alpha \\ge 0, \\, \\beta \\ge 0$, and $\\alpha + \\beta < 1$.

# 

# The innovation distribution $F\_Z(z)$ is modeled via EVT:

# $$F\_Z(z) = \\begin{cases} 

# \\tau\_L \\left\[1 + \\frac{\\xi\_L (|z| - |u\_L|)}{\\beta\_L}\\right]^{-1/\\xi\_L} \& \\text{for } z < u\_L \\\\

# \\hat{F}\_{\\text{emp}}(z) \& \\text{for } u\_L \\le z \\le u\_R \\\\

# 1 - \\tau\_R \\left\[1 + \\frac{\\xi\_R (z - u\_R)}{\\beta\_R}\\right]^{-1/\\xi\_R} \& \\text{for } z > u\_R 

# \\end{cases}$$

# 

# \#### 3. Calibration \& Parameter Estimation

# 1\. Fit $(\\omega, \\alpha, \\beta)$ by minimizing the negative log-likelihood of centered returns:

# &#x20;  $$\\mathcal{L}\_{GARCH}(\\omega, \\alpha, \\beta) = \\frac{1}{2} \\sum\_{t=1}^N \\left\[ \\ln(\\sigma\_t^2) + \\frac{(R\_t - \\bar{R})^2}{\\sigma\_t^2} \\right]$$

# 2\. Extract standardized innovations:

# &#x20;  $$z\_t = \\frac{R\_t - \\bar{R}}{\\sigma\_t}, \\quad t \\in \[1, N]$$

# 3\. Fit GPD shape and scale parameters to the extreme 5th and 95th percentiles of the standardized residuals $\\{z\_t\\}$.

# 

# \#### 4. Valuation \& Simulation Process

# For each path $i$:

# 1\. Initialize variance at current market state: $\\sigma\_1^2 = \\sigma\_{\\text{latest}}^2$, and $S\_1 = S\_0$.

# 2\. For each day $t \\in \\{1, \\dots, D\\}$:

# &#x20;  \* Draw an EVT-standardized shock $Z\_t$ using Inverse Transform Sampling.

# &#x20;  \* Compute dynamic physical drift: $\\mu\_t = \\left(\\mu - q - \\frac{1}{2}(\\sigma\_t^2 \\cdot 252)\\right) \\Delta t$.

# &#x20;  \* Calculate daily log return: $r\_t = \\mu\_t + \\sigma\_t Z\_t$.

# &#x20;  \* Update spot price: $S\_{t+1} = S\_t \\exp(r\_t)$.

# &#x20;  \* Update next-day conditional variance: $\\sigma\_{t+1}^2 = \\omega + \\alpha (\\sigma\_t Z\_t)^2 + \\beta \\sigma\_t^2$.

# 3\. Compute discounted terminal payoffs across all simulated paths.

# 

# \---

# 

# \### Model 7: MAD-Based Filtered Historical Simulation ($L\_1$ Norm + EVT)

# 

# \#### 1. Conceptual Intuition

# Standard GARCH(1,1) relies on squared shocks ($\\epsilon\_{t-1}^2$, an $L\_2$ norm). For fat-tailed asset returns where the tail index $\\alpha \\le 4$, the theoretical fourth moment is infinite:

# $$\\mathbb{E}\[R^4] = \\infty$$

# Under these conditions, squaring extreme returns destabilizes calibration, causing sample variance estimators to explode unreliably. 

# 

# Model 7 replaces the fragile $L\_2$ variance filter with an $L\_1$ Mean Absolute Deviation (MAD) Exponentially Weighted Moving Average (EWMA) filter. Because the first moment of financial returns is always finite ($\\alpha > 1$), standardizing returns by the $L\_1$ norm yields robust residuals that are declustered and modeled via EVT without numerical instability.

# 

# \#### 2. Probabilistic Notation \& Distributional Assumptions

# The daily price process follows:

# $$R\_t = \\mu\_t + \\text{MAD}\_t \\cdot z\_t, \\quad z\_t \\stackrel{\\text{i.i.d.}}{\\sim} F\_{\\text{MAD-EVT}}$$

# The $L\_1$ scale parameter $\\text{MAD}\_t = \\mathbb{E}\[|R\_t - \\mathbb{E}\[R\_t]|]$ updates via EWMA with decay factor $\\lambda = 0.94$:

# $$\\text{MAD}\_t = \\lambda \\text{MAD}\_{t-1} + (1 - \\lambda)|R\_{t-1}|$$

# Standardized residuals $z\_t = \\frac{R\_t}{\\text{MAD}\_t}$ are declustered via the Runs Method ($k=5$) to isolate independent shock peaks, and tail exceedances are fitted to a Generalized Pareto Distribution.

# 

# \#### 3. Calibration \& Parameter Estimation

# 1\. Compute the historical recursive MAD series:

# &#x20;  $$\\text{MAD}\_1 = \\frac{1}{N}\\sum\_{k=1}^N |R\_k|, \\quad \\text{MAD}\_t = \\lambda \\text{MAD}\_{t-1} + (1 - \\lambda)|R\_{t-1}|$$

# 2\. Extract $L\_1$ standardized residuals:

# &#x20;  $$z\_t = \\frac{R\_t}{\\text{MAD}\_t}$$

# 3\. Establish 5th and 95th percentile thresholds on $\\{z\_t\\}$. Apply the Runs Method ($k=5$) to extract declustered residual exceedances.

# 4\. Fit GPD parameters $(\\xi\_L, \\beta\_L)$ and $(\\xi\_R, \\beta\_R)$ to residual exceedances via MLE.

# 

# \#### 4. Valuation \& Simulation Process

# For each path $i$:

# 1\. Initialize $\\text{MAD}\_1 = \\text{MAD}\_{\\text{latest}}$, and $S\_1 = S\_0$.

# 2\. For each day $t \\in \\{1, \\dots, D\\}$:

# &#x20;  \* Draw an EVT residual $z\_t$ from the spliced declustered residual distribution.

# &#x20;  \* Compute the simulated physical return component: $r\_{\\text{sim}} = z\_t \\cdot \\text{MAD}\_t$.

# &#x20;  \* Calculate the local drift using the calibrated annualized scale correction:

# &#x20;    $$\\mu\_t = \\left(\\mu - q - \\frac{1}{2}(\\text{MAD}\_t^2 \\cdot 252)\\right) \\Delta t$$

# &#x20;  \* Update underlying price: $S\_{t+1} = S\_t \\exp(r\_{\\text{sim}} + \\mu\_t)$.

# &#x20;  \* Update the $L\_1$ scale parameter recursively for tomorrow:

# &#x20;    $$\\text{MAD}\_{t+1} = \\lambda \\text{MAD}\_t + (1 - \\lambda)|r\_{\\text{sim}}|$$

# 3\. Discount the terminal intrinsic payoffs back to $t\_0$:

# &#x20;  $$\\mathbb{E}\[V] = e^{-rT} \\frac{1}{M} \\sum\_{i=1}^M \\text{Payoff}(S\_T^{(i)})$$

# 

# \---

# 

# \## 🔬 Mathematical Parameters \& Statistical Interpretations

# 

# | Parameter | Mathematical Domain | Theoretical Description |

# | :--- | :--- | :--- |

# | \*\*Tail Shape ($\\xi$)\*\* | $\\xi \\in (-\\infty, \\infty)$ | Governs tail decay rate. $\\xi \\le 0$ implies thin/Gaussian decay. $0 < \\xi < 0.5$ indicates heavy, Paretian tails. $\\xi \\ge 0.5$ implies infinite variance ($\\mathbb{E}\[X^2] = \\infty$). Tail index is given by $\\alpha\_{\\text{tail}} = 1/\\xi$. |

# | \*\*Tail Scale ($\\beta$)\*\* | $\\beta > 0$ | Measures the dispersion and magnitude of exceedances above the threshold $u$. |

# | \*\*GARCH $\\omega$\*\* | $\\omega > 0$ | The long-term baseline variance intercept: $\\sigma^2\_{\\infty} = \\frac{\\omega}{1 - (\\alpha + \\beta)}$. |

# | \*\*GARCH $\\alpha$\*\* | $\\alpha \\ge 0$ | The shock reaction coefficient (sensitivity of conditional variance to the previous day's squared innovation $\\epsilon\_{t-1}^2$). |

# | \*\*GARCH $\\beta$\*\* | $\\beta \\ge 0$ | The persistence coefficient (memory of past volatility regimes). |

# | \*\*Persistence ($\\alpha+\\beta$)\*\* | $\\alpha + \\beta < 1$ | The rate of mean-reversion toward long-term variance. Values $> 0.98$ represent severe volatility clustering. |

# | \*\*MAD Decay ($\\lambda$)\*\* | $\\lambda \\in (0, 1)$ | Controls the exponential memory horizon of the $L\_1$ MAD filter. Set to $\\lambda = 0.94$ (effective memory horizon of $\\sim \\frac{1}{1-\\lambda} \\approx 17$ trading days). |

# | \*\*Runs Window ($k$)\*\* | $k \\in \\mathbb{N}^+$ | Separation window for the Runs Method declustering algorithm ($k=5$ trading days). Eliminates dependent intra-cluster peaks. |

# 

# \---

# 

# \## 📊 Analytical Greeks \& Risk Profiling Engine

# 

# The application includes an analytical Greek calculation engine supporting first-order, second-order, and cross-sensitivities across multi-leg structures:

# 

# \### First-Order Greeks

# \* \*\*Delta ($\\Delta$):\*\* First derivative of structure value with respect to spot price:

# &#x20; $$\\Delta = \\frac{\\partial V}{\\partial S} = \\sum\_{j=1}^M w\_j \\phi\_j e^{-q T} \\mathcal{N}(\\phi\_j d\_{1,j})$$

# \* \*\*Vega ($\\nu$):\*\* Sensitivity to a $1\\%$ absolute move in implied volatility:

# &#x20; $$\\nu = \\frac{1}{100} \\frac{\\partial V}{\\partial \\sigma} = \\frac{1}{100} \\sum\_{j=1}^M w\_j S\_0 e^{-q T} \\phi\_j n(d\_{1,j}) \\sqrt{T}$$

# \* \*\*Theta ($\\Theta$):\*\* Calendar day time decay ($\\frac{1}{365} \\frac{\\partial V}{\\partial t}$):

# &#x20; $$\\Theta\_{\\text{call}} = \\frac{1}{365}\\left\[ -\\frac{S\_0 n(d\_1)\\sigma e^{-q T}}{2\\sqrt{T}} - r K e^{-r T} \\mathcal{N}(d\_2) + q S\_0 e^{-q T} \\mathcal{N}(d\_1) \\right]$$

# &#x20; $$\\Theta\_{\\text{put}} = \\frac{1}{365}\\left\[ -\\frac{S\_0 n(d\_1)\\sigma e^{-q T}}{2\\sqrt{T}} + r K e^{-r T} \\mathcal{N}(-d\_2) - q S\_0 e^{-q T} \\mathcal{N}(-d\_1) \\right]$$

# \* \*\*Rho ($\\rho$):\*\* Sensitivity to a $1\\%$ change in the risk-free interest rate:

# &#x20; $$\\rho = \\frac{1}{100} \\frac{\\partial V}{\\partial r} = \\frac{1}{100} \\sum\_{j=1}^M w\_j \\phi\_j K\_j T e^{-r T} \\mathcal{N}(\\phi\_j d\_{2,j})$$

# 

# \### Second-Order \& Cross Greeks

# \* \*\*Gamma ($\\Gamma$):\*\* Curvature and directional acceleration:

# &#x20; $$\\Gamma = \\frac{\\partial^2 V}{\\partial S^2} = \\sum\_{j=1}^M w\_j \\frac{e^{-q T} n(d\_{1,j})}{S\_0 \\sigma \\sqrt{T}}$$

# \* \*\*Vomma (Volga):\*\* Volatility convexity (rate of change of Vega with respect to volatility):

# &#x20; $$\\text{Vomma} = \\frac{\\partial^2 V}{\\partial \\sigma^2} = \\nu \\cdot \\frac{d\_{1} d\_{2}}{\\sigma}$$

# \* \*\*Vanna:\*\* Cross-derivative of option value with respect to spot price and volatility:

# &#x20; $$\\text{Vanna} = \\frac{\\partial^2 V}{\\partial S \\partial \\sigma} = -e^{-q T} n(d\_1) \\frac{d\_2}{\\sigma}$$

# 

# \---

# 

# \## 🛠️ Installation \& Setup

# 

# \### Prerequisites

# \* Python 3.10+

# \* Git

# 

# \### Local Deployment

# 1\. Clone the repository:

# &#x20;  ```bash

# &#x20;  git clone \[https://github.com/yourusername/option-valuation-tool.git](https://github.com/yourusername/option-valuation-tool.git)

# &#x20;  cd option-valuation-tool

