use std::iter::IntoIterator;

use statrs::distribution::{ContinuousCDF, Normal};

use crate::{Computation, Error, Float};

/// Performs the Anderson-Darling test for normality.
///
/// The Anderson-Darling test is a modification of the
/// Kolmogorov-Smirnov test that gives more weight to the tails of the distribution.
///
/// Takes one argument `data` which is an iterator over floating-point numbers ([`impl
/// IntoIterator<Item = T>`](IntoIterator)).
///
/// The sample size of `data` must be greater than or equal to 8.
/// Also, the range of `data` must not be equal to 0.
///
/// # Examples
///
/// ```
/// use normality::anderson_darling;
///
/// let normal_data = vec![-1.1, 0.2, -0.4, 0.0, -0.7, 1.2, -0.1, 0.8, 0.5, -0.9];
/// let result = anderson_darling(normal_data).unwrap();
/// // p-value should be high for normal data
/// assert!(result.p_value > 0.05);
///
/// let uniform_data = vec![2.0, 1.0, 0.9, 1.0, 2.0, 2.0, 2.0, 2.0];
/// let result_uniform = anderson_darling(uniform_data).unwrap();
/// // p-value should be low for non-normal data
/// assert!(result_uniform.p_value < 0.05);
/// ```
pub fn anderson_darling<T: Float, I: IntoIterator<Item = T>>(
    data: I,
) -> Result<Computation<T>, Error> {
    let data: Vec<T> = data.into_iter().collect();
    let n = data.len();
    if n < 8 {
        return Err(Error::InsufficientSampleSize {
            given: n,
            needed: 8,
        });
    }

    if data.iter().any(|&v| v.is_nan()) {
        return Err(Error::ContainsNaN);
    }

    let mut sorted_data = data;
    sorted_data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let n_t = T::from(n).unwrap();
    let sum = sorted_data.iter().fold(T::zero(), |acc, &x| acc + x);
    let mean = sum / n_t;

    let variance =
        sorted_data.iter().map(|&x| (x - mean).powi(2)).fold(T::zero(), |acc, x| acc + x)
            / T::from(n - 1).unwrap();

    let std_dev = variance.sqrt();

    if std_dev < T::epsilon() {
        return Err(Error::ZeroRange);
    }

    let standard_normal = Normal::new(0.0, 1.0)?;

    let logp1: Vec<T> = sorted_data
        .iter()
        .map(|&x| {
            let z = (x - mean) / std_dev;
            T::from(standard_normal.cdf(z.to_f64().unwrap())).unwrap().ln()
        })
        .collect();

    let logp2: Vec<T> = sorted_data
        .iter()
        .map(|&x| {
            let z = (x - mean) / std_dev;
            T::from(standard_normal.cdf(-z.to_f64().unwrap())).unwrap().ln()
        })
        .collect();

    let h_sum: T = (0..n)
        .map(|i| {
            let i_t = T::from(i + 1).unwrap();
            let factor = T::from(2.0).unwrap() * i_t - T::one();
            let term = logp1[i] + logp2[n - 1 - i];
            factor * term
        })
        .fold(T::zero(), |acc, val| acc + val);

    let a = -n_t - h_sum / n_t;
    let aa = (T::one() + T::from(0.75).unwrap() / n_t + T::from(2.25).unwrap() / n_t.powi(2)) * a;

    let p_value = if aa < T::from(0.2).unwrap() {
        T::one()
            - (-T::from(13.436).unwrap() + T::from(101.14).unwrap() * aa
                - T::from(223.73).unwrap() * aa.powi(2))
            .exp()
    } else if aa < T::from(0.34).unwrap() {
        T::one()
            - (-T::from(8.318).unwrap() + T::from(42.796).unwrap() * aa
                - T::from(59.938).unwrap() * aa.powi(2))
            .exp()
    } else if aa < T::from(0.6).unwrap() {
        (T::from(0.9177).unwrap()
            - T::from(4.279).unwrap() * aa
            - T::from(1.38).unwrap() * aa.powi(2))
        .exp()
    } else if aa < T::from(10.0).unwrap() {
        (T::from(1.2937).unwrap() - T::from(5.709).unwrap() * aa
            + T::from(0.0186).unwrap() * aa.powi(2))
        .exp()
    } else {
        T::from(3.7e-24).unwrap()
    };

    Ok(Computation {
        statistic: a,
        p_value: p_value.max(T::zero()).min(T::one()),
    })
}
