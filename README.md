# normality

[![Crates.io Version](https://img.shields.io/crates/v/normality)](https://crates.io/crates/normality)
[![Documentation](https://docs.rs/normality/badge.svg)](https://docs.rs/normality)
[![License](https://img.shields.io/crates/l/normality.svg)](./LICENSE)

A Rust crate for assessing the normality of a data sample. It provides several common statistical tests to determine if a set of data is likely drawn from a normal distribution.

All test implementations are generic and can work with `f32` or `f64` data types. The implementations are ported from well-established algorithms found in popular R packages.

## Implemented Tests

### Univariate Normality
- [Shapiro-Wilk Test](https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test)
- [Lilliefors (Kolmogorov-Smirnov) Test](https://en.wikipedia.org/wiki/Lilliefors_test)
- [Anderson-Darling Test](https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test)
- [Jarque-Bera Test](https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test)
- [Pearson Chi-squared Test](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test)
- [D'Agostino's K-squared Test](https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test)
- [Anscombe-Glynn Kurtosis Test](https://doi.org/10.1093/biomet/70.1.227)
- [Energy Test](https://www.sciencedirect.com/science/article/pii/S0378375820301154)

### Multivariate Normality
- [Henze-Zirkler Test](https://doi.org/10.1080/03610929008830400)
- [Henze-Wagner Test](https://doi.org/10.1006/jmva.1997.1684)
- [Mardia's Test](https://doi.org/10.2307/2334770)
- [Pudelko's Test](https://web.archive.org/web/20220620071059/https://www.math.uni.wroc.pl/~pms/files/25.1/Article/25.1.3.pdf)

## Installation
Either run `cargo add normality` or add the crate to your `Cargo.toml`:

```toml
[dependencies]
normality = "2"

# To enable parallel execution for faster performance on large data:
# normality = { version = "2", features = ["parallel"] }
```

## Example Usage

### Univariate
```rust
use normality::{shapiro_wilk, Error};

fn main() -> Result<(), Error> {
    // Sample data that is likely from a normal distribution
    let data = vec![-1.1, -0.8, -0.5, -0.2, 0.0, 0.2, 0.5, 0.8, 1.1, 1.3];

    // Perform the Shapiro-Wilk test
    let result = shapiro_wilk(data)?;

    println!("Shapiro-Wilk Test Results:");
    println!("  W-statistic: {:.4}", result.statistic);
    println!("  p-value: {:.4}", result.p_value);

    // Interpretation: A high p-value (e.g., > 0.05) suggests that the data
    // does not significantly deviate from a normal distribution.
    if result.p_value > 0.05 {
        println!("Conclusion: The sample is likely from a normal distribution.");
    } else {
        println!("Conclusion: The sample is not likely from a normal distribution.");
    }

    Ok(())
}
```

### Multivariate
#### Using [`vec!`](https://doc.rust-lang.org/std/macro.vec.html)
```rust
use nalgebra::matrix;
use normality::multivariate::{henze_zirkler, HenzeZirklerMethod};
use normality::Error;

fn main() -> Result<(), Error> {
    // 3D data from a multivariate normal distribution
    let data = vec![
        vec![0.1, 0.2, 0.3],
        vec![0.5, 0.1, 0.4],
        vec![-0.2, 0.3, 0.1],
        vec![0.0, 0.0, 0.0],
        vec![0.8, -0.5, 0.2],
        vec![-0.1, -0.1, -0.1],
    ];

    // Perform the Henze-Zirkler test
    let result = henze_zirkler(data, false, HenzeZirklerMethod::LogNormal)?;

    println!("Henze-Zirkler Test Results:");
    println!("  HZ-statistic: {:.4}", result.statistic);
    println!("  p-value: {:.4}", result.p_value);

    if result.p_value > 0.05 {
        println!("Conclusion: The sample is likely from a multivariate normal distribution.");
    }

    Ok(())
}
```

#### Using [`nalgebra::matrix!`](https://docs.rs/nalgebra/latest/nalgebra/macro.matrix.html)
```rust
use nalgebra::matrix;
use normality::multivariate::{henze_zirkler, HenzeZirklerMethod};
use normality::Error;

fn main() -> Result<(), Error> {
    // 3D data from a multivariate normal distribution
    let data = matrix![0.1, 0.2, 0.3;
        0.5, 0.1, 0.4;
        -0.2, 0.3, 0.1;
        0.0, 0.0, 0.0;
        0.8, -0.5, 0.2;
        -0.1, -0.1, -0.1];

    // Perform the Henze-Zirkler test
    let result = henze_zirkler(data.row_iter().map(|row| row.into_iter().copied()), false, HenzeZirklerMethod::LogNormal)?;

    println!("Henze-Zirkler Test Results:");
    println!("  HZ-statistic: {:.4}", result.statistic);
    println!("  p-value: {:.4}", result.p_value);

    if result.p_value > 0.05 {
        println!("Conclusion: The sample is likely from a multivariate normal distribution.");
    }

    Ok(())
}
```

## Parallelism
This crate supports optional parallelism via the [`rayon`](https://crates.io/crates/rayon) crate. This can significantly improve performance for large datasets by parallelizing sorting and statistical calculations.

To enable parallelism, add the `parallel` feature to your `Cargo.toml`:

```toml
[dependencies]
normality = { version = "2", features = ["parallel"] }
```

When enabled, functions will automatically use parallel iterators and parallel sorting algorithms. No changes to your code are required.

## Accuracy
The accuracy of the implemented tests has been verified against their R equivalents. Running the integration tests for this crate requires a local installation of R and for the `Rscript` executable to be available in the system's PATH. 

## License
This project is licensed under the MIT License.
