use std::f64::consts::SQRT_2;

use nalgebra::{DMatrix, SymmetricEigen};
use rand::distributions::Distribution;
use rand::thread_rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use statrs::distribution::Normal;
use statrs::function::gamma::ln_gamma;
use xsf::hyp1f1;

use crate::{Computation, Error, Float};

fn mvnorm_e_statistic(x: &DMatrix<f64>) -> f64 {
    let n = x.nrows();
    let d = x.ncols();
    let mut mean = DMatrix::zeros(1, d);

    for i in 0..n {
        mean += x.row(i);
    }

    mean /= n as f64;
    let mut z = DMatrix::zeros(n, d);

    for i in 0..n {
        z.set_row(i, &(x.row(i) - &mean));
    }

    let cov = (&z.transpose() * &z) / ((n - 1) as f64);
    let sym_cov = SymmetricEigen::new(cov);
    let p = sym_cov.eigenvectors;
    let lambda = sym_cov.eigenvalues;
    let mut d_inv_sqrt = DMatrix::zeros(d, d);

    for i in 0..d {
        if lambda[i] > 1e-12 {
            d_inv_sqrt[(i, i)] = 1.0 / lambda[i].sqrt();
        }
    }

    let s_inv_half = &p * d_inv_sqrt * p.transpose();
    let y = &z * s_inv_half;
    let d_f64 = d as f64;
    let n_f64 = n as f64;
    let const_val = (ln_gamma(f64::midpoint(d_f64, 1.0)) - ln_gamma(d_f64 / 2.0)).exp();
    let mean2 = 2.0 * const_val;
    let mut sum_mean1 = 0.0;

    for i in 0..n {
        let ysq = y.row(i).norm_squared();
        sum_mean1 += hyp1f1(-0.5, d_f64 / 2.0, -ysq / 2.0);
    }

    let mean1 = SQRT_2 * const_val * (sum_mean1 / n_f64);
    let mut sum_dist = 0.0;

    for i in 0..n {
        for j in (i + 1)..n {
            let diff = y.row(i) - y.row(j);
            sum_dist += diff.norm();
        }
    }

    let mean3 = (2.0 * sum_dist) / (n_f64 * n_f64);

    n_f64 * (2.0 * mean1 - mean2 - mean3)
}

/// Performs the Székely-Rizzo test for multivariate normality.
///
/// This is an energy distance-based goodness-of-fit test. It measures the distance
/// between the empirical distribution of the sample's scaled residuals and a standard
/// multivariate normal distribution. It is particularly powerful against heavy-tailed
/// alternatives.
///
/// Takes an argument `data` which is an iterator of iterators representing the dataset (rows are
/// observations).
///
/// Also takes `replicates`, which is the number of parametric bootstrap (Monte Carlo) replicates
/// to use for estimating the p-value. If `replicates` is `0`, only the baseline test statistic
/// is computed and the p-value will be `NaN`.
///
/// # Examples
///
/// ```
/// use normality::multivariate::szekely_rizzo;
///
/// // 3D data from a multivariate normal distribution
/// let data = vec![
///     vec![0.1, 0.2, 0.3],
///     vec![0.5, 0.1, 0.4],
///     vec![-0.2, 0.3, 0.1],
///     vec![0.0, 0.0, 0.0],
///     vec![0.8, -0.5, 0.2],
///     vec![-0.1, -0.1, -0.1],
/// ];
///
/// // Run test with 100 parametric bootstrap replicates
/// let result = szekely_rizzo(data, 100).unwrap();
/// assert!(result.statistic > 0.5);
/// ```
pub fn szekely_rizzo<T: Float, I: IntoIterator<Item = J>, J: IntoIterator<Item = T>>(data: I, replicates: usize) -> Result<Computation<T>, Error> {
    let mut n = 0;
    let mut d = None;
    let mut flat_data = Vec::new();

    for row in data {
        let mut cols = 0;
        for val in row {
            let val_f64 = val.to_f64().unwrap_or(f64::NAN);

            if val_f64.is_nan() {
                return Err(Error::ContainsNaN);
            }

            flat_data.push(val_f64);
            cols += 1;
        }

        if let Some(expected_cols) = d {
            if cols != expected_cols {
                return Err(Error::DimensionMismatch);
            }
        } else {
            d = Some(cols);
        }

        n += 1;
    }

    let d = d.unwrap_or(0);

    if n < 2 {
        return Err(Error::InsufficientSampleSize {
            given: n,
            needed: 2,
        });
    }

    let x = DMatrix::from_row_slice(n, d, &flat_data);
    let t0 = mvnorm_e_statistic(&x);
    let p_value = if replicates == 0 {
        f64::NAN
    } else {
        let normal = Normal::new(0.0, 1.0)?;

        #[cfg(feature = "parallel")]
        let greater_count: usize = (0..replicates)
            .into_par_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let sim_data: Vec<f64> = (0..(n * d)).map(|_| normal.sample(&mut rng)).collect();
                let sim_matrix = DMatrix::from_row_slice(n, d, &sim_data);

                usize::from(mvnorm_e_statistic(&sim_matrix) >= t0)
            })
            .sum();

        #[cfg(not(feature = "parallel"))]
        let greater_count: usize = (0..replicates)
            .into_iter()
            .map(|_| {
                let mut rng = thread_rng();
                let sim_data: Vec<f64> = (0..(n * d)).map(|_| normal.sample(&mut rng)).collect();
                let sim_matrix = DMatrix::from_row_slice(n, d, &sim_data);

                usize::from(mvnorm_e_statistic(&sim_matrix) >= t0)
            })
            .sum();

        (greater_count as f64 + 1.0) / (replicates as f64 + 1.0)
    };

    Ok(Computation {
        statistic: T::from(t0).unwrap(),
        p_value: T::from(p_value).unwrap(),
    })
}
