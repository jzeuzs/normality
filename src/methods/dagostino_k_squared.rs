use std::iter::IntoIterator;

use statrs::distribution::{ContinuousCDF, Normal};

use crate::{Computation, Error, Float};

/// Performs D'Agostino's K-squared test for skewness to assess normality.
///
/// The test evaluates the null hypothesis that the sample's skewness
/// is the same as that of a normal distribution (i.e., zero).
///
/// The test first calculates the sample skewness, then transforms it into a `z` statistic
/// that, under the null hypothesis, follows a standard normal distribution. This implementation
/// performs a two-sided test.
///
/// Takes one argument `data` which is an iterator over floating-point numbers ([`impl
/// IntoIterator<Item = T>`](IntoIterator)).
///
/// The sample size of `data` must be between 8 and 46840.
/// Also, the range of `data` must not be equal to 0.
///
/// # Examples
///
/// ```
/// use normality::dagostino_k_squared;
///
/// let normal_data = vec![-1.1, 0.2, -0.4, 0.0, -0.7, 1.2, -0.1, 0.8, 0.5, -0.9];
/// let result = dagostino_k_squared(normal_data).unwrap();
/// // p-value should be high for normal data
/// assert!(result.p_value > 0.05);
///
/// let uniform_data =
///     vec![2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
/// let result_uniform = dagostino_k_squared(uniform_data).unwrap();
/// // p-value should be low for non-normal data
/// assert!(result_uniform.p_value < 0.05);
/// ```
pub fn dagostino_k_squared<T: Float, I: IntoIterator<Item = T>>(
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

    if n > 46340 {
        return Err(Error::ExcessiveSampleSize {
            given: n,
            needed: 46340,
        });
    }

    if data.iter().any(|&v| v.is_nan()) {
        return Err(Error::ContainsNaN);
    }

    let n_t = T::from(n).unwrap();
    let mean = data.iter().copied().sum::<T>() / n_t;

    let deviations: Vec<T> = data.iter().map(|&x| x - mean).collect();
    let sum_sq_devs = deviations.iter().map(|&d| d.powi(2)).sum::<T>();

    if sum_sq_devs < T::epsilon() {
        return Err(Error::ZeroRange);
    }

    // Calculate skewness (s3)
    let m3 = deviations.iter().map(|&d| d.powi(3)).sum::<T>() / n_t;
    let m2 = sum_sq_devs / n_t;
    let s3 = m3 / m2.powf(T::from(1.5).unwrap());

    // --- Transformation logic from R's fBasics::agostino.test ---
    let y = s3
        * ((n_t + T::one()) * (n_t + T::from(3.0).unwrap())
            / (T::from(6.0).unwrap() * (n_t - T::from(2.0).unwrap())))
        .sqrt();

    let n_sq = n_t * n_t;
    let b2_num = T::from(3.0).unwrap()
        * (n_sq + T::from(27.0).unwrap() * n_t - T::from(70.0).unwrap())
        * (n_t + T::one())
        * (n_t + T::from(3.0).unwrap());

    let b2_den = (n_t - T::from(2.0).unwrap())
        * (n_t + T::from(5.0).unwrap())
        * (n_t + T::from(7.0).unwrap())
        * (n_t + T::from(9.0).unwrap());

    let b2 = b2_num / b2_den;

    let w_sq = (T::from(2.0).unwrap() * (b2 - T::one())).sqrt() - T::one();
    let w = w_sq.sqrt();
    let d = T::one() / w.ln().sqrt();
    let a = (T::from(2.0).unwrap() / (w_sq - T::one())).sqrt();

    let y_over_a = y / a;
    let z = d * (y_over_a + (y_over_a.powi(2) + T::one()).sqrt()).ln();

    // Calculate two-sided p-value from the standard normal distribution
    let normal_dist = Normal::new(0.0, 1.0)?;
    let mut pval = T::from(2.0).unwrap() * T::from(normal_dist.sf(z.to_f64().unwrap())).unwrap();

    if pval > T::one() {
        pval = T::from(2.0).unwrap() - pval;
    }

    Ok(Computation {
        statistic: z,
        p_value: pval,
    })
}
