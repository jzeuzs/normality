---
title: "normality: A Rust crate for univariate and multivariate normality testing"

tags:
  - Rust
  - statistics
  - normality
  - normality-test
  - hypothesis-testing

authors:
  - name: Jezzu Morrisen C. Quimosing
    orcid: 0009-0001-4302-7946
    affiliation: 1

affiliations:
  - name: School of Statistics, University of the Philippines Diliman
    index: 1
    ror: "03tbh6y23"

date: 20 March 2026
bibliography: paper.bib
---

# Summary
The assumption of normality is a common prerequisite for numerous parametric statistical tests, from $t$-tests to complex linear regression models. Validating this assumption is of importance. As discussed by @ferr_normal_2025, real-world data often violates this assumption, exhibiting skewness, kurtosis, and outliers. Moreover, errors regarding the distribution of data are remarkably common in published scientific literature [@ghasemi_normality_2012].

`normality` is a Rust library designed to address this by providing a suite of hypothesis tests for assessing both univariate and multivariate normality. By implementing widely used tests such as the Shapiro-Wilk, Anderson-Darling, Jarque-Bera, and Henze-Zirkler tests, `normality` provides researchers with the tools to validate the distribution of their data within the rapidly growing Rust ecosystem.

# Statement of need
Scientists and researchers alike are increasingly turning to Rust for scientific computing due to its C-like performance and guaranteed memory safety [@perkel_why_2020, @bugden_rust_2022]. Recent studies in fields such as computational physics show that rewriting C++ simulations in Rust can yield performance increases while making parallelization safer and simpler to implement [@veytsman_rewrite_2024]. As Rust expands heavily into big data engineering and data-intensive applications [@chilukoori_role_2024], there now is a need for statistical validation tools native to this ecosystem.

Currently, while researchers can visually assess distributions, further validation requires formal hypothesis testing. The misuse of these tests is prevalent; for instance, a review by @midway_testing_2025 found that over 70% of ecology papers and 90% of biology papers incorrectly applied normality tests to raw data instead of regression model residuals. A well-documented Rust crate can help guide developers toward better and sound methodological practices.

Furthermore, literature indicates that no single normality test is universally the most powerful across all sample sizes and underlying distributions [@arnastauskaite_exhaustive_2021]. Hence, a library like `normality`, which has multiple methods rather than a single approach, is needed for data science. It allows users to integrate the most appropriate checks into data pipelines and workflows where manual inspection is difficult.

# State of the field
The Rust ecosystem for data science and statistics has seen large growth over the years, establishing the groundwork for a complete scientific suite as documented in recent literature covering statistics with Rust [@nakamura_statistics_2024]. Several libraries currently exist:

- `polars` [@vink_polars_2020] provides a fast and multithreaded query engine for DataFrames;
- `statrs` [@ma_statrs_2016] provides a set of statistical utilities, implementations of common distributions, and mathematical functions;
- `polyfit` [@carson_polyfit_2026] handles linear regression and curve fitting, also provides a collection of statistical tests to validate fit quality; and
- `augurs` [@sully_augurs_2024] is a time series analysis and forecasting toolkit.

While these tools excel in their respective domains, there remains a gap in hypothesis testing for data distributions. `normality` fills this gap. It complements `statrs` by providing the formal tests needed to see if data actually follows the theoretical distributuions. Further, it also serves as a utility tool for libraries like `polyfit` and `augurs`, which can utilize `normality`'s tests to validate model residuals—helping prevent the methodological errors discussed by @midway_testing_2025.

# Software design
The `normality` crate is designed to be modular, extensible, and mathematically accurate and rigorous. The implemented tests are the following:

+---------------------------------------+----------------+-------------------------------------+
| Name                                  | Classification | Reference                           |
|                                       |                |                                     |
+:=====================================:+:==============:+:===================================:+
| Shapiro-Wilk                          | Univariate     | @shapiro_analysis_1965              |
+---------------------------------------+----------------+-------------------------------------+
| Lilliefors (Kolmogorov-Smirnov)       | Univariate     | @lilliefors_kolmogorov-smirnov_1967 |
+---------------------------------------+----------------+-------------------------------------+
| Anderson-Darling                      | Univariate     | @anderson_test_1954                 |
+---------------------------------------+----------------+-------------------------------------+
| Jarque-Bera                           | Univariate     | @jarque_test_1987                   |
+---------------------------------------+----------------+-------------------------------------+
| Pearson Chi-squared                   | Univariate     | @pearson_criterion_1900             |
+---------------------------------------+----------------+-------------------------------------+
| D'Agostino's K-squared                | Univariate     | @dagostino_suggestion_1990          |
+---------------------------------------+----------------+-------------------------------------+
| Anscombe-Glynn                        | Univariate     | @anscombe_distribution_1983         |
+---------------------------------------+----------------+-------------------------------------+
| Energy Test                           | Univariate     | @mori_energy_2021                   |
+---------------------------------------+----------------+-------------------------------------+
| Henze-Zirkler                         | Multivariate   | @henze_class_1990                   |
+---------------------------------------+----------------+-------------------------------------+
| Henze-Wagner                          | Multivariate   | @henze_new_1997                     |
+---------------------------------------+----------------+-------------------------------------+
| Mardia                                | Multivariate   | @mardia_measures_1970               |
+---------------------------------------+----------------+-------------------------------------+
| Pudelko                               | Multivariate   | @pudelko_new_2005                   |
+---------------------------------------+----------------+-------------------------------------+

The above tests are regularly tested for accuracy with their R equivalents.

# AI usage disclosure

Google's Gemini was used by the author solely in order to check the grammatical accuracy of this paper. No generative AI tools were used in the development of the library. 

# Acknowledgments
We acknowledge the R Core Team [@r_core_team_r_2025] and the authors of the R packages `nortest` [@gross_nortest_2006] and `moments` [@komsta_moments_2005], from which the algorithms of this crate were inspired from. We also acknowledge the authors of the python package `scipy` [@virtanen_scipy_2020] for their Cython implementation of the Shapiro-Wilk test.

# References
