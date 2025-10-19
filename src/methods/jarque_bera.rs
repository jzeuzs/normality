use statrs::distribution::{ChiSquared, ContinuousCDF};

use crate::{Computation, Error, Float};

/// Performs the Jarque-Bera test for normality.
///
/// The test determines whether sample
/// data have skewness and kurtosis matching a normal distribution.
///
/// The test statistic is calculated based on the sample's skewness and excess kurtosis.
/// Under the null hypothesis of normality, this statistic follows a chi-squared
/// distribution with 2 degrees of freedom.
///
/// Takes one argument `data` which is a slice (`&[T]`) containing the data sample. This can be a
/// [`Vec<T>`](Vec), an array [`[T; N]`](std::slice), or an [`ndarray::Array1<T>`](ndarray::Array1).
///
/// The sample size of `data` must be greater than or equal to 3.
/// Also, the range of `data` must not be equal to 0.
///
/// # Examples
///
/// ```
/// use normality::jarque_bera;
///
/// let normal_data = vec![-1.1, 0.2, -0.4, 0.0, -0.7, 1.2, -0.1, 0.8, 0.5, -0.9];
/// let result = jarque_bera(&normal_data).unwrap();
/// // p-value should be high for normal data
/// assert!(result.p_value > 0.05);
///
/// let uniform_data =
///     vec![2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
/// let result_uniform = jarque_bera(&uniform_data).unwrap();
/// // p-value should be low for non-normal data
/// assert!(result_uniform.p_value < 0.05);
/// ```
pub fn jarque_bera<T: Float>(data: &[T]) -> Result<Computation<T>, Error> {
    let n = data.len();
    if n < 3 {
        return Err(Error::InsufficientSampleSize {
            needed: 3,
            given: n,
        });
    }

    if data.iter().any(|&v| v.is_nan()) {
        return Err(Error::ContainsNaN);
    }

    let n_t = T::from(n).unwrap();

    let m1 = data.iter().fold(T::zero(), |acc, &x| acc + x) / n_t; // Mean

    let deviations: Vec<T> = data.iter().map(|&x| x - m1).collect();

    let m2 = deviations.iter().map(|&d| d.powi(2)).sum::<T>() / n_t; // Variance (biased)

    if m2 < T::epsilon() {
        return Err(Error::ZeroRange);
    }

    let m3 = deviations.iter().map(|&d| d.powi(3)).sum::<T>() / n_t;
    let m4 = deviations.iter().map(|&d| d.powi(4)).sum::<T>() / n_t;

    let skewness = m3 / m2.powf(T::from(1.5).unwrap());
    let kurtosis = m4 / m2.powi(2);

    let s_sq = skewness.powi(2);
    let k_minus_3_sq = (kurtosis - T::from(3.0).unwrap()).powi(2);

    // Formula is equivalent to n/6 * S^2 + n/24 * (K-3)^2
    let statistic = (n_t / T::from(6.0).unwrap()) * (s_sq + T::from(0.25).unwrap() * k_minus_3_sq);

    // The p-value is the survival function of the chi-squared distribution with 2 degrees of
    // freedom.
    let chi_squared_dist = ChiSquared::new(2.0)?;
    let p_value = T::from(chi_squared_dist.sf(statistic.to_f64().unwrap())).unwrap();

    Ok(Computation {
        statistic,
        p_value,
    })
}
