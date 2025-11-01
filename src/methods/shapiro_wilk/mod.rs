mod swilk;

use std::iter::IntoIterator;

use ndarray::Array1;
use swilk::swilk;

use crate::{Computation, Error, Float};

/// Performs the Shapiro-Wilk test for normality on a given sample of data.
///
/// The Shapiro-Wilk test is a statistical test of the null hypothesis that a sample
/// came from a normally distributed population.
///
/// This function is generic and can operate on types `T` supported by [`num_traits::Float`].
///
/// Takes one argument `data` which is an iterator over floating-point numbers ([`impl
/// IntoIterator<Item = T>`](IntoIterator)).
///
/// The sample size of `data` must be between 3 and 5000.
/// Also, the range of `data` must not be equal to 0.
///
/// # Examples
///
/// ```
/// use normality::{Computation, Error, shapiro_wilk};
///
/// // Example with a sample that should be normal
/// let normal_data: [f64; 10] = [1.2, 0.8, 1.5, 0.9, 1.0, 1.1, 0.7, 1.3, 1.4, 0.6];
///
/// let result = shapiro_wilk(normal_data).unwrap();
///
/// assert!(result.p_value > 0.05); // Expect to not reject the null hypothesis
///
/// // Example with a sample that is clearly not normal
/// let uniform_data: [f32; 20] = [
///     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0,
///     10.0, 10.0, 10.0,
/// ];
///
/// let result_uniform = shapiro_wilk(uniform_data).unwrap();
///
/// assert!(result_uniform.p_value < 0.05); // Expect to reject the null hypothesis
///
/// // Example of handling an error
/// let small_data = [1.0, 2.0];
/// let error_result = shapiro_wilk(small_data);
///
/// assert_eq!(
///     error_result,
///     Err(Error::InsufficientSampleSize {
///         given: 2,
///         needed: 3
///     })
/// );
/// ```
pub fn shapiro_wilk<T: Float, I: IntoIterator<Item = T>>(data: I) -> Result<Computation<T>, Error> {
    let data: Vec<T> = data.into_iter().collect();
    let n = data.len();

    if n < 3 {
        return Err(Error::InsufficientSampleSize {
            given: n,
            needed: 3,
        });
    }

    let mut a = Array1::<T>::zeros(n / 2);
    let init = false;
    let n1_in = -1;

    let mut y_vec = data;
    y_vec.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    let mut y = Array1::from_vec(y_vec);

    let median_val = y[n / 2];
    y.mapv_inplace(|v| v - median_val);

    let (w, pw, ifault) = swilk(y.view(), a.view_mut(), init, n1_in);

    if ifault != 0 {
        return match ifault {
            2 => Err(Error::ExcessiveSampleSize {
                given: n,
                needed: 5000,
            }),
            6 => Err(Error::ZeroRange),
            _ => Err(Error::Other(format!("Internal error with ifault {ifault}"))),
        };
    }

    let result = Computation {
        statistic: w,
        p_value: pw,
    };

    Ok(result)
}
