---
title: 'normtest: A rust crate assessing the normality of a data sample'

tags:
  - Rust
  - statistics
  - normality
  - normality-test

authors:
  - name: Jezzu Morrisen C. Quimosing
    orcid: 0009-0001-4302-7946
    affiliation: 1

affiliations:
  - name: Independent Researcher, Philippines
    index: 1

date: 19 October 2025
bibliography: paper.bib
---

# Summary
Assessing distributional assumptions is an important step in statistical inference and modeling. Amongst these, the assumption of normality is common and support the validity of many parametric methods. The `normality` crate offers a collection of tools for conducting goodness-of-fit tests against a normal distribution in the Rust programming language.

While using a normality test to select between subsequent parametric and non-parametric tests can affect the soundness of further inference unless the $p$-value of the final test is adjusted for the pre-testing step, formal hypothesis tests for normality provide utility in several other context [@rochon_test_2012]. It may serve as a complement to graphical methods such as Q-Q plots for exploratory data analysis and for the diagnostic checking of residuals in regression and other statistical models.

# Statement of need
The Rust programming language is gaining traction in fields such as data engineering and scientific computing due to its performance and memory safety [@chilukoori_role_nodate; @harmouch_rust_2023]. While a foundational ecosystem for statistics in Rust is emerging, with general-purpose libraries such as `statrs` [@ma_statrs_2016], `rs-stats` [@lsh0x_rs-stats_2022], and `hypors` [@agrawal_hypors_2024], there still remains a need tools that focus on specific statistical domains.

The `normality` crate fills this gap by proving a single and reliable crate that consolidates different widely-used normality testing methods into one package. Each test implemented has been tested for accuracy against its equivalent in the R programming language [@r_core_team_r_2025].

THe implemented tests are the following:

- Shapiro-Wilk Test [@shapiro_analysis_1965];
- Lilliefors (Kolmogorov-Smirnow) Test [@lilliefors_kolmogorov-smirnov_1967];
- Anderson-Darling Test [@anderson_test_1954];
- Jarque-Bera Test [@jarque_test_1987];
- D'Agostino's K-squared Test [@dagostino_suggestion_1990];
- Cramer-von Mises Test [@cramer_composition_1928]; and
- Pearson Chi-squared Test [@pearson_x_1900].

# Mathematics
The tests implemented in `normality` rely on established statistical formulas. These tests can be categorized based on their approach.

Moment-based tests, such as the Jarque-Bera test, use sample skewness ($S$) and kurtosis ($K$) to check for normality. The test statistic for a sample of size $n$ is:

$$ JB = \frac{n}{6} \left(S^2 + \frac{(K-3)^2}{4}  \right) $$

Under the null hypothesis, the $JB$ statistic is asymptotically distributed as a chi-squared distribution with two degrees of freedom.

Likewise, Empirical Distribution Function (EDF) tests, like the Anderson-Darling test, measure the discrepancy between the standardized empirical distribution of the sample and the theoretical normal distribution. The Anderson-Darling statistic ($A^2$) is a weighted squared difference, giving more weight to the tails of the distribution:

$$ A^2 = -n - \frac{1}{n} \sum_{i=1}^n (2i-1) [\ln (\Phi (Z_i)) + \ln (1 - \Phi (Z_{n - i + 1}))] $$

where $Z_i$ are the ordered, standardized observations and $\Phi$ is the standard normal cumulative distribution function.

Another category of test, which is exemplified by the Shapiro-Wilk test, is based on the correlation between the sample data and the expected order statistics from a normal distribution. It calculates the $W$ statistic, which is a ratio of two estimates of the population variance:

$$ W = \frac{\left(\displaystyle \sum_{i=1}^n a_i x_{(i)} \right)^2}{ \displaystyle \sum_{i=1}^n (x_i - \overline{x})^2} $$

Here, $x_{(i)}$ are the ordered sample values, $\overline{x}$ is the sample mean, and the coefficients $a_i$ are derived from the expected values of the order statistics of a sample from a standard normal distribution.

# Acknowledgments
We acknowledge the R Core Team [@r_core_team_r_2025] and the authors of the R packages `nortest` [@gross_nortest_2006] and `moments` [@komsta_moments_2005], from which the algorithms of this crate were inspired from. We also acknowledge the authors of the python package `scipy` [@virtanen_scipy_2020] for their Cython implementation of the Shapiro-Wilk test.

# References
