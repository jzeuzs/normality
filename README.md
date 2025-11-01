# normality

A Rust crate for assessing the normality of a data sample. It provides several common statistical tests to determine if a set of data is likely drawn from a normal distribution.

All test implementations are generic and can work with `f32` or `f64` data types. The implementations are ported from well-established algorithms found in popular R packages.

## Implemented Tests
- [Shapiro-Wilk Test](https://en.wikipedia.org/wiki/Shapiro%E2%80%93Wilk_test)
- [Lilliefors (Kolmogorov-Smirnov) Test](https://en.wikipedia.org/wiki/Lilliefors_test)
- [Anderson-Darling Test](https://en.wikipedia.org/wiki/Anderson%E2%80%93Darling_test)
- [Jarque-Bera Test](https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test)
- [Pearson Chi-squared Test](https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test)
- [Cramer-von Mises Test](https://en.wikipedia.org/wiki/Cram%C3%A9r%E2%80%93von_Mises_criterion)
- [D'Agostino's K-squared Test](https://en.wikipedia.org/wiki/D%27Agostino%27s_K-squared_test)
- [Anscombe-Glynn Kurtosis Test](https://doi.org/10.1093/biomet/70.1.227)

## Installation
Either run `cargo add normality` or add the crate to your `Cargo.toml`:

```toml
[dependencies]
normality = "1.0.0"
```

## Example Usage
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

## Accuracy
The accuracy of the implemented tests has been verified against their R equivalents. Running the integration tests for this crate requires a local installation of R and for the `Rscript` executable to be available in the system's PATH.

## License
This project is licensed under the MIT License.
