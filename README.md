# Law of Large Numbers (LLN) & Central Limit Theorem (CLT) Simulator

An interactive Streamlit dashboard designed to rigorously visualize the statistical behavior of probability distributions across the thin-tailed to fat-tailed spectrum. 

This simulator demonstrates the physical mechanics of statistical convergence—and, critically, the *failure* of convergence in extreme fat-tailed domains (like Pareto or Cauchy distributions). It includes empirical diagnostic tools used to detect when standard statistical assumptions collapse.

## Mathematical Foundations & Core Concepts

At a fundamental level, probability and statistics rely heavily on the hidden assumptions of **stationarity** and finite moments. This simulator stress-tests what happens when those mathematical prerequisites are broken.

### The Law of Large Numbers (LLN)

**Intuitive Definition:** If you repeatedly sample from a stable environment, the average of your observations will eventually lock onto the true mathematical average of that environment. As your sample size grows, the noise of individual random events cancels out.

**Mathematical Definition:** Let $X_1, X_2, \dots, X_n$ be a sequence of random variables with a true expected value $\mu = E[X]$. Let $\bar{X}_n$ be the sample mean:

$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i
$$

The Weak Law of Large Numbers states that for any margin of error $\epsilon > 0$, the probability that the sample mean deviates from the true mean approaches zero as $n \to \infty$:

$$
\lim_{n \to \infty} P(\vert{}\bar{X}_n - \mu\vert{} \ge \epsilon) = 0
$$

**Strict Requirements for the LLN:**
* **Finite First Moment (**$E[\vert{}X\vert{}] < \infty$**):** The mathematical expectation of the absolute value of the distribution must be finite. If the mean is undefined (e.g., Cauchy distribution), the sample average will endlessly jump when extreme outliers arrive and will never converge.
* **Identically Distributed (Stationarity):** The variables must come from the exact same probability distribution.
* **Independence:** The variables must not influence one another. 

**Convergence of Specific Sample Statistics:**

| Statistic | Does it Converge? | Requirement / Boundary Condition | 
| ----- | ----- | ----- | 
| **Sample average** | Yes | Requires a finite mean. | 
| **Sample median** | Yes | Requires the CDF to be strictly increasing at the median. | 
| **Sample percentiles** | Yes | Guaranteed by the Glivenko-Cantelli theorem. | 
| **Sample mean absolute dev.** | Yes | Requires a finite mean. | 
| **Sample standard deviation** | Yes | Requires finite variance. | 
| **Sample skewness & kurtosis**| **Conditional**| Higher moments require finite moments up to order $k$. They break down incredibly fast in fat-tailed environments. |
| **Sample min & max** | **No** | Governed by Extreme Value Theory, not the LLN. | 

### The Breakdown of Higher-Order Moments

The simulator allows tracking of the 3rd moment (Sample Skewness) and 4th moment (Sample Kurtosis). These higher-order moments break down much faster than the mean. 

A distribution only possesses finite, calculable statistical moments up to $k$, where $k < \alpha$ (the tail exponent). 
*   If you select a **Student-t with df=2.5**, the mean and variance are finite and will visibly converge on the LLN chart. However, if you switch the tracked statistic to **Sample Skewness ($p=3$)**, the LLN will violently jump off the chart when outliers appear. The moment is mathematically undefined, rendering standard skewness metrics completely invalid for this dataset.

### The Central Limit Theorem (CLT)

While the LLN dictates *where* the sample mean heads, the CLT dictates the *shape* of the errors around that mean.

**Mathematical Definition:** Let $X_1, X_2, \dots, X_n$ be a sequence of independent and identically distributed (i.i.d.) random variables with a true expected value $\mu = E[X]$ and a strictly finite variance $\sigma^2 = Var(X) < \infty$. 

The Average of Sample Variables ($\bar{X}_n = \frac{S_n}{n}$) converges to a Normal distribution centered on the true mean. Its variance shrinks proportionally to $n$:

$$
\bar{X}_n \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n}\right)
$$

The Standardized Average ($Z_n$) converges exactly in distribution ($\xrightarrow{d}$) to the Standard Normal distribution:

$$
Z_n = \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0,1)
$$

**Strict Requirements for the CLT:**
* **Finite Variance (**$\sigma^2 < \infty$**):** The core mathematical prerequisite. The variance dictates the scaling factor ($\sigma / \sqrt{n}$) in the formula.
* **Finite Mean (**$\mu$ **exists):** Centering the data is impossible without a defined mean.

**Convergence of Specific Sample Statistics:**

| Statistic | Does it Converge? | Limiting Distribution / Boundary Condition | 
| ----- | ----- | ----- | 
| **Sample average** | Yes | Normal Distribution ($\mathcal{N}$). Requires finite variance. | 
| **Sample median** | Yes | Normal Distribution ($\mathcal{N}$). Requires a positive density at the median. | 
| **Sample percentiles** | Yes | Normal Distribution ($\mathcal{N}$). Governed by the Bahadur representation. | 
| **Sample mean absolute dev.** | Yes | Normal Distribution ($\mathcal{N}$). Requires finite variance. | 
| **Sample standard deviation** | Yes | Normal Distribution ($\mathcal{N}$). Requires a strictly finite 4th moment (kurtosis). | 
| **Sample skewness & kurtosis**| **Conditional**| Normal Distribution ($\mathcal{N}$). Skewness requires a finite 6th moment. Kurtosis requires a finite 8th moment. These fail almost universally in real-world data. |
| **Sample min & max** | **No** | Converges to the Generalized Extreme Value (GEV) distribution (Fréchet, Gumbel, or Weibull), never Gaussian. | 

### The Generalized Central Limit Theorem & Pre-Asymptotic Convergence

When the strict requirement of finite variance is broken ($\sigma^2 = \infty$), the standard CLT collapses. If the tails decay as a power law (e.g., Pareto distribution where $\alpha < 2$), the sum of the variables is governed by the **Generalized Central Limit Theorem**, converging instead to a **Lévy Alpha-Stable Distribution** retaining infinite variance.

Furthermore, we must account for the **Speed of Convergence (The Slow Law of Large Numbers)**. For fat-tailed distributions, convergence is agonizingly slow. To match the statistical stability of a sample mean derived from 1,000 Gaussian observations:
* **Borderline Fat Tail (**$\alpha = 2$**):** Requires roughly **10,000 observations**.
* **Standard Financial Tail (**$\alpha = 1.5$**):** Requires roughly **1,000,000 observations**.
* **Extreme Fat Tail (**$\alpha = 1.15$**):** Requires a sample size exceeding **100 trillion observations**.

**Empirical Examples of Alpha ($\alpha$) in the Real World:**
Because we only have a few decades of daily data (roughly 10,000 days), we are permanently stuck in the pre-asymptotic domain for many real-world phenomena.
*   **Broad Stock Indices & Macroeconomic Variables:** $\alpha \approx 3$ (e.g., S&P 500, GDP drops, Inflation rates).
*   **Individual Stocks:** $\alpha \approx 1.5$ to $2.5$.
*   **Wars (Casualties):** $\alpha \approx 1.3$ to $1.8$.
*   **Pandemics (Fatalities) & Extreme Operational Risk:** $\alpha \approx 0.5$ to $1.2$.

## Empirical Fat Tail Diagnostics

When dealing with finite real-world data, theoretical asymptotic limits (like the LLN) are not enough. We must empirically diagnose the "fatness" of the tails to understand what statistical rules apply. This simulator includes three robust visual heuristics to detect tail behavior:

**1. Zipf Plot (Log-Log Survival Function)**
Instead of a standard cumulative distribution, this plots the threshold $K$ against the probability of exceeding it, $P(X > K)$, on a double-logarithmic scale.
*   **Thin Tails:** The curve will drop almost vertically, proving extreme outliers face a strict probabilistic ceiling.
*   **Fat Tails:** The data will form a straight, slowly decaying, downward-sloping line. The slope of this line is the negative tail exponent ($-\alpha$), visually proving that extreme events scale according to a power law.

**2. Maximum-to-Sum Plot**
A direct, visual stress-test of the Law of Large Numbers for moments $p=1, 2, 3, 4$. It plots sample size $n$ against the ratio of the single maximum observation to the sum of all observations: $\frac{\max(|X|^p)}{\sum |X|^p}$.
*   **Thin Tails:** The maximum is swallowed by the sum. All lines ($p=1, 2, 3, 4$) smoothly converge to 0.
*   **Fat Tails:** For undefined moments, the line will hover above zero or jump erratically. If the $p=2$ line refuses to drop, a single outlier is dominating the entire sample variance, proving the variance is mathematically infinite.

**3. Mean Excess Plot (Excess Conditional Expectation)**
Measures the expected severity of a deviation *given* that a threshold $K$ has already been breached: $e(K) = E[|X| - K \mid |X| > K]$.
*   **Thin Tails:** Slopes downward. Gravity pulls extremes back to the mean.
*   **Fat Tails:** Slopes upward. Extremes accelerate—the worse an event gets, the worse it is expected to get.

## Supported Distributions & Tail Behavior

All distributions in this simulator are mathematically shifted to a central mean of `0` (where a mean exists) to allow for direct visual comparability.

**1. Discrete & Thin-Tailed (Binomial, Poisson, Geometric):** 
LLN & CLT apply. Convergence is generally fast. Gaussian smoothing happens rapidly.

**2. Continuous Thin-Tailed (Normal, Exponential, Weibull):** 
LLN & CLT apply. Convergence ranges from instantaneous (Normal) to moderate.

**3. Fat-Tailed (Student-t, Pareto):** 
*   If variance exists, CLT applies, but convergence is agonizingly slow (the pre-asymptotic domain). 
*   If variance is infinite but mean exists (e.g., Pareto with $\alpha$ between 1 and 2, Student-t with df between 1 and 2), the standard CLT fails completely. The LLN applies but converges too slowly to be practically relevant. 

*(Specific Highlights in the Simulator)*
*   **The 80/20 Principle (Pareto $\alpha = 1.16$):** This specific parameter mathematically generates the famous "80/20 Rule" (where 80% of the effects come from 20% of the causes). Simulating this proves that in such an environment, the mean is so unstable that it is practically meaningless in finite samples.
*   **Borderline Fractional Tails (Student-t df = 2.25 to 2.75):** These fractional degrees of freedom perfectly mimic real-world financial assets (like individual equities). They live in a dangerous borderline zone: they possess a finite variance (allowing the CLT to *eventually* work), but their pre-asymptotic convergence is so agonizingly slow that standard Gaussian metrics will severely underprice risk in any measurable human timeframe.

**4. The Unruly Distribution (Cauchy):** 
Neither the mean nor the variance exists. Averages wander erratically forever. Both LLN and standard CLT fail completely.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/lln-clt-simulator.git
   cd lln-clt-simulator
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## Usage Guide

1.  **Select a Distribution:** Start with a `Normal (Thin Tail)` distribution, then switch to an extreme fat-tailed distribution like `Pareto, α=1.16 (80/20 Principle)` to witness the breakdown of the LLN.
2.  **Select a Statistic:** Toggle between tracking the Sample Mean, Sample Variance, or specific Percentiles (including Higher-Order moments like Skewness and Kurtosis).
3.  **Set Sample Sizes:** Adjust the $n$ for the CLT trials and the total $n$ limit for the LLN cumulative path.
4.  **Freeze the Chart Bounds:** Manually lock the Y-axis constraints (e.g., `-1` to `1`). This is critical for seeing massive outliers blast past the expected limits of the chart when examining fat-tailed behavior.

## Dependencies

*   [Streamlit](https://streamlit.io/): Web framework and UI.
*   [SciPy](https://scipy.org/): Core random variable generation and statistical functions.
*   [NumPy](https://numpy.org/): High-performance vectorization and array manipulation.
*   [Plotly](https://plotly.com/python/): Interactive charting.
*   [Pandas](https://pandas.pydata.org/): Data structuring for Plotly.