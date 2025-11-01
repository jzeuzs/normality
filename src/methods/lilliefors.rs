use std::iter::IntoIterator;

use statrs::distribution::{ContinuousCDF, Normal};

use crate::{Computation, Error, Float};

/// Performs the Lilliefors (Kolmogorov-Smirnov) test for normality.
///
/// The test determines if a sample comes from a normally
/// distributed population when the mean and variance are unknown.
///
/// Takes one argument `data` which is an iterator over floating-point numbers ([`impl
/// IntoIterator<Item = T>`](IntoIterator)).
///
/// The sample size of `data` must be greater than or equal to 5.
/// Also, the range of `data` must not be equal to 0.
///
/// # Examples
///
/// ```
/// use normality::lilliefors;
///
/// let normal_data = vec![-1.1, 0.2, -0.4, 0.0, -0.7, 1.2, -0.1, 0.8, 0.5, -0.9];
/// let result = lilliefors(normal_data).unwrap();
/// // p-value should be high for normal data
/// assert!(result.p_value > 0.05);
///
/// let uniform_data = vec![2.0, 1.0, 0.9, 1.0, 2.0, 2.0, 2.0, 2.0];
/// let result_uniform = lilliefors(uniform_data).unwrap();
/// // p-value should be low for non-normal data
/// assert!(result_uniform.p_value < 0.05);
/// ```
pub fn lilliefors<T: Float, I: IntoIterator<Item = T>>(data: I) -> Result<Computation<T>, Error> {
    let data: Vec<T> = data.into_iter().collect();
    let n = data.len();

    if n < 5 {
        return Err(Error::InsufficientSampleSize {
            given: n,
            needed: 5,
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
    let p_values: Vec<T> = sorted_data
        .iter()
        .map(|&x| {
            let z = (x - mean) / std_dev;
            T::from(standard_normal.cdf(z.to_f64().unwrap())).unwrap()
        })
        .collect();

    let d_plus = (0..n)
        .map(|i| T::from((i + 1) as f64 / n as f64).unwrap() - p_values[i])
        .fold(T::neg_infinity(), num_traits::Float::max);

    let d_minus = (0..n)
        .map(|i| p_values[i] - T::from(i as f64 / n as f64).unwrap())
        .fold(T::neg_infinity(), num_traits::Float::max);

    let k = d_plus.max(d_minus);

    // P-value approximation logic ported from R's nortest::lillie.test
    let (kd, nd) = if n <= 100 {
        (k, T::from(n).unwrap())
    } else {
        (k * (n_t / T::from(100.0).unwrap()).powf(T::from(0.49).unwrap()), T::from(100.0).unwrap())
    };

    let mut p_value = (-T::from(7.01256).unwrap() * kd.powi(2) * (nd + T::from(2.78019).unwrap())
        + T::from(2.99587).unwrap() * kd * (nd + T::from(2.78019).unwrap()).sqrt()
        - T::from(0.122_119).unwrap()
        + T::from(0.974_598).unwrap() / nd.sqrt()
        + T::from(1.67997).unwrap() / nd)
        .exp();

    if p_value > T::from(0.1).unwrap() {
        let kk = (n_t.sqrt() - T::from(0.01).unwrap() + T::from(0.85).unwrap() / n_t.sqrt()) * k;

        p_value = if kk <= T::from(0.302).unwrap() {
            T::one()
        } else if kk <= T::from(0.5).unwrap() {
            T::from(2.76773).unwrap() - T::from(19.828_315).unwrap() * kk
                + T::from(80.709_644).unwrap() * kk.powi(2)
                - T::from(138.55152).unwrap() * kk.powi(3)
                + T::from(81.218_052).unwrap() * kk.powi(4)
        } else if kk <= T::from(0.9).unwrap() {
            -T::from(4.901_232).unwrap() + T::from(40.662_806).unwrap() * kk
                - T::from(97.490_286).unwrap() * kk.powi(2)
                + T::from(94.029_866).unwrap() * kk.powi(3)
                - T::from(32.355_711).unwrap() * kk.powi(4)
        } else if kk <= T::from(1.31).unwrap() {
            T::from(6.198_765).unwrap() - T::from(19.558_097).unwrap() * kk
                + T::from(23.186_922).unwrap() * kk.powi(2)
                - T::from(12.234_627).unwrap() * kk.powi(3)
                + T::from(2.423_045).unwrap() * kk.powi(4)
        } else {
            T::zero()
        };
    }

    Ok(Computation {
        statistic: k,
        p_value: p_value.max(T::zero()).min(T::one()), // Ensure p-value is in [0, 1]
    })
}
