#[cfg(feature = "parallel")]
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

use crate::{Computation, Error, Float};

/// Performs the Anscombe-Glynn kurtosis test for normality.
///
/// The test first calculates the sample kurtosis, then transforms it into a `z` statistic
/// using the Wilson-Hilferty transformation, which, under the null hypothesis,
/// follows a standard normal distribution. This implementation performs a two-sided test.
///
/// Takes one argument `data` which is an iterator over floating-point numbers ([`impl
/// IntoIterator<Item = T>`](IntoIterator)).
///
/// The sample size of `data` must be greater than or equal to 4.
/// Also, the range of `data` must not be equal to 0.
///
/// # Examples
///
/// ```
/// use normality::anscombe_glynn;
///
/// let normal_data = vec![-1.1, 0.2, -0.4, 0.0, -0.7, 1.2, -0.1, 0.8, 0.5, -0.9];
/// let result = anscombe_glynn(normal_data).unwrap();
/// // p-value should be high for normal data
/// assert!(result.p_value > 0.05);
///
/// let uniform_data = vec![2.0, 1.0, 0.9, 1.0, 2.0, 2.0, 2.0, 2.0];
/// let result_uniform = anscombe_glynn(uniform_data).unwrap();
/// // p-value should be low for non-normal data
/// assert!(result_uniform.p_value < 0.05);
/// ```
pub fn anscombe_glynn<T: Float, I: IntoIterator<Item = T>>(
    data: I,
) -> Result<Computation<T>, Error> {
    let data_vec: Vec<T> = data.into_iter().collect();
    let n = data_vec.len();
    if n < 4 {
        // The formulas involve n-2 and n-3 in the denominator.
        return Err(Error::InsufficientSampleSize {
            given: n,
            needed: 4,
        });
    }

    if data_vec.iter().any(|v| v.is_nan()) {
        return Err(Error::ContainsNaN);
    }

    let n_t = T::from(n).unwrap();
    let n_sq = n_t * n_t;

    let mean = iter_if_parallel!(&data_vec).copied().sum::<T>() / n_t;

    #[cfg(feature = "parallel")]
    let (sum_sq_devs, sum_fourth_devs) = data_vec
        .par_iter()
        .map(|&x| {
            let dev = x - mean;
            (dev.powi(2), dev.powi(4))
        })
        .reduce(|| (T::zero(), T::zero()), |a, b| (a.0 + b.0, a.1 + b.1));

    #[cfg(not(feature = "parallel"))]
    let (sum_sq_devs, sum_fourth_devs) =
        data_vec.iter().fold((T::zero(), T::zero()), |(sum_sq, sum_fourth), &x| {
            let dev = x - mean;
            (sum_sq + dev.powi(2), sum_fourth + dev.powi(4))
        });

    if sum_sq_devs < T::epsilon() {
        return Err(Error::ZeroRange);
    }

    // Kurtosis statistic (b)
    let b = n_t * sum_fourth_devs / sum_sq_devs.powi(2);

    // Expected value of b (eb2)
    let eb2 = T::from(3.0).unwrap() * (n_t - T::one()) / (n_t + T::one());

    // Variance of b (vb2)
    let vb2_num = T::from(24.0).unwrap()
        * n_t
        * (n_t - T::from(2.0).unwrap())
        * (n_t - T::from(3.0).unwrap());

    let vb2_den =
        (n_t + T::one()).powi(2) * (n_t + T::from(3.0).unwrap()) * (n_t + T::from(5.0).unwrap());

    let vb2 = vb2_num / vb2_den;

    // Skewness of b (m3)
    let m3_term1 = (T::from(6.0).unwrap()
        * (n_sq - T::from(5.0).unwrap() * n_t + T::from(2.0).unwrap()))
        / ((n_t + T::from(7.0).unwrap()) * (n_t + T::from(9.0).unwrap()));

    let m3_term2_num =
        T::from(6.0).unwrap() * (n_t + T::from(3.0).unwrap()) * (n_t + T::from(5.0).unwrap());

    let m3_term2_den = n_t * (n_t - T::from(2.0).unwrap()) * (n_t - T::from(3.0).unwrap());
    let m3 = m3_term1 * (m3_term2_num / m3_term2_den).sqrt();
    let m3_sq = m3 * m3;

    // Wilson-Hilferty transformation parameters (a, d)
    // Here `a` is the shape parameter.
    let a_term = T::from(2.0).unwrap() / m3;
    let a = T::from(6.0).unwrap()
        + (T::from(8.0).unwrap() / m3)
            * (a_term + (T::one() + T::from(4.0).unwrap() / m3_sq).sqrt());

    // Standardized kurtosis (xx)
    let xx = (b - eb2) / vb2.sqrt();

    // Transformed statistic (z)
    let z_num = T::one()
        - T::from(2.0).unwrap() / (T::from(9.0).unwrap() * a)
        - ((T::one() - T::from(2.0).unwrap() / a)
            / (T::one() + xx * (T::from(2.0).unwrap() / (a - T::from(4.0).unwrap())).sqrt()))
        .powf(T::one() / T::from(3.0).unwrap());

    let z_den = (T::from(2.0).unwrap() / (T::from(9.0).unwrap() * a)).sqrt();
    let z = z_num / z_den;

    // Calculate two-sided p-value from the standard normal distribution
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    let mut pval = T::from(2.0).unwrap() * T::from(normal_dist.sf(z.to_f64().unwrap())).unwrap();

    if pval > T::one() {
        pval = T::from(2.0).unwrap() - pval;
    }

    Ok(Computation {
        statistic: z,
        p_value: pval,
    })
}
