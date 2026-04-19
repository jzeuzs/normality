#![allow(clippy::unreadable_literal)]

use std::cmp::Ordering;

use nalgebra::{DMatrix, DVector, SymmetricEigen};
use rand::distributions::Distribution;
use rand::thread_rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

use crate::{Computation, Error, Float};

/// Specifies the method for p-value calculation in Royston's test.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum RoystonMethod {
    /// Use the asymptotic Chi-Squared approximation.
    Asymptotic,
    /// Use a parametric bootstrap (Monte Carlo) simulation with a given number of replicates.
    MonteCarlo(usize),
}

/// Computes the standard Pearson kurtosis for a slice of data
fn pearson_kurtosis(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let sum_m2: f64 = x.iter().map(|&val| (val - mean).powi(2)).sum();
    let sum_m4: f64 = x.iter().map(|&val| (val - mean).powi(4)).sum();

    (n * sum_m4) / (sum_m2 * sum_m2)
}

/// Computes the Shapiro-Francia W' statistic for leptokurtic data
fn shapiro_francia_statistic(x: &[f64]) -> f64 {
    let mut sorted = x.to_vec();
    sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

    let n = sorted.len();
    let norm = Normal::new(0.0, 1.0).unwrap();
    let mut y = Vec::with_capacity(n);

    for i in 0..n {
        let p = ((i + 1) as f64 - 0.375) / (n as f64 + 0.25);
        y.push(norm.inverse_cdf(p));
    }

    let mean_x = sorted.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;
    let mut sum_dx2 = 0.0;
    let mut sum_dy2 = 0.0;
    let mut sum_dxdy = 0.0;

    for i in 0..n {
        let dx = sorted[i] - mean_x;
        let dy = y[i] - mean_y;

        sum_dx2 += dx * dx;
        sum_dy2 += dy * dy;
        sum_dxdy += dx * dy;
    }

    if sum_dx2 == 0.0 || sum_dy2 == 0.0 {
        return 1.0;
    }

    let cor = sum_dxdy / (sum_dx2.sqrt() * sum_dy2.sqrt());
    cor * cor
}

fn royston_h_statistic(x: &DMatrix<f64>) -> Result<(f64, f64), Error> {
    let n = x.nrows();
    let p = x.ncols();
    let n_f64 = n as f64;
    let p_f64 = p as f64;
    let mean = x.row_mean();
    let mut centered = x.clone();

    for mut row in centered.row_iter_mut() {
        row -= &mean;
    }

    let cov = (&centered.transpose() * &centered) / (n_f64 - 1.0);
    let mut std_devs = DVector::zeros(p);

    for i in 0..p {
        std_devs[i] = cov[(i, i)].sqrt();

        if std_devs[i] == 0.0 {
            return Err(Error::ZeroRange);
        }
    }

    let mut cor = DMatrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            if i == j {
                cor[(i, j)] = 1.0;
            } else {
                let cij = cov[(i, j)] / (std_devs[i] * std_devs[j]).clamp(-1.0, 1.0);
                cor[(i, j)] = cij;
            }
        }
    }

    let is_small = n <= 11;
    let lx = n_f64.ln();
    let g = if is_small { -2.273 + 0.459 * n_f64 } else { 0.0 };
    let m = if is_small {
        0.544 - 0.39978 * n_f64 + 0.025054 * n_f64.powi(2) - 0.0006714 * n_f64.powi(3)
    } else {
        -1.5861 - 0.31082 * lx - 0.083751 * lx.powi(2) + 0.0038915 * lx.powi(3)
    };

    let s = if is_small {
        (1.3822 - 0.77857 * n_f64 + 0.062767 * n_f64.powi(2) - 0.0020322 * n_f64.powi(3)).exp()
    } else {
        (-0.4803 - 0.082676 * lx + 0.0030302 * lx.powi(2)).exp()
    };

    let mut z_vals = vec![0.0; p];
    for (j, z_val) in z_vals.iter_mut().enumerate().take(p) {
        let col: Vec<f64> = x.column(j).iter().copied().collect();
        let kurt = pearson_kurtosis(&col);

        let mut w = if kurt > 3.0 {
            shapiro_francia_statistic(&col)
        } else {
            crate::shapiro_wilk(col)?.statistic
        };

        if w >= 1.0 {
            w = 1.0 - 1e-16;
        }

        if is_small {
            *z_val = (-(g - (1.0 - w).ln()).ln() - m) / s;
        } else {
            *z_val = ((1.0 - w).ln() - m) / s;
        }
    }

    let u = 0.715;
    let v = 0.21364 + 0.015124 * lx.powi(2) - 0.0018034 * lx.powi(3);
    let mut sum_nc = 0.0;

    for i in 0..p {
        for j in 0..p {
            let cij = cor[(i, j)];
            let nc = cij.powi(5) * (1.0 - (u * (1.0 - cij).powf(u)) / v);
            sum_nc += nc;
        }
    }

    let t = sum_nc - p_f64;
    let mc = t / (p_f64.powi(2) - p_f64);
    let edf = p_f64 / (1.0 + (p_f64 - 1.0) * mc);
    let norm = Normal::new(0.0, 1.0).unwrap();
    let mut sum_res = 0.0;

    for z in z_vals {
        let p_val = norm.cdf(-z).max(1e-16);
        let q_val = norm.inverse_cdf(p_val / 2.0);
        sum_res += q_val * q_val;
    }

    let h_stat = (edf * sum_res) / p_f64;

    Ok((h_stat, edf))
}

/// Performs Royston's test for multivariate normality.
///
/// This test combines univariate Shapiro-Wilk (or Shapiro-Francia for leptokurtic data)
/// statistics across variables and adjusts for their correlation structure.
///
/// Takes an argument `data` which is an iterator of iterators representing the dataset (rows are
/// observations).
///
/// Also takes `method`, which is the method for calculating the p-value
/// ([`Asymptotic`](RoystonMethod::Asymptotic) or
/// [`MonteCarlo`](RoystonMethod::MonteCarlo)).
///
/// # Examples
///
/// ```
/// use normality::multivariate::{RoystonMethod, royston};
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
/// // Run asymptotic test
/// let result = royston(data, RoystonMethod::Asymptotic).unwrap();
/// assert!(result.p_value > 0.05);
/// ```
pub fn royston<T: Float, I: IntoIterator<Item = J>, J: IntoIterator<Item = T>>(
    data: I,
    method: RoystonMethod,
) -> Result<Computation<T>, Error> {
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

    let p = d.unwrap_or(0);

    if n <= 3 {
        return Err(Error::InsufficientSampleSize {
            given: n,
            needed: 4,
        });
    }

    if n > 2000 {
        return Err(Error::ExcessiveSampleSize {
            given: n,
            needed: 2000,
        });
    }

    if p < 2 {
        return Err(Error::Other("Royston's test requires at least 2 variables.".to_string()));
    }

    let x = DMatrix::from_row_slice(n, p, &flat_data);
    let (h_stat, edf) = royston_h_statistic(&x)?;
    let p_value = match method {
        RoystonMethod::Asymptotic => {
            let chisq = ChiSquared::new(edf)?;
            1.0 - chisq.cdf(h_stat)
        },
        RoystonMethod::MonteCarlo(replicates) => {
            let mean = x.row_mean();
            let mut centered = x.clone();
            for mut row in centered.row_iter_mut() {
                row -= &mean;
            }

            let cov = (&centered.transpose() * &centered) / ((n - 1) as f64);
            let sym_cov = SymmetricEigen::new(cov);
            let p_mat = sym_cov.eigenvectors;
            let lambda = sym_cov.eigenvalues;

            let mut d_sqrt = DMatrix::zeros(p, p);
            for i in 0..p {
                if lambda[i] > 0.0 {
                    d_sqrt[(i, i)] = lambda[i].sqrt();
                }
            }

            let sigma_half = &p_mat * d_sqrt * p_mat.transpose();
            let normal = Normal::new(0.0, 1.0)?;

            #[cfg(feature = "parallel")]
            let greater_count: usize = (0..replicates)
                .into_par_iter()
                .map(|_| {
                    let mut rng = thread_rng();
                    let sim_data: Vec<f64> =
                        (0..(n * p)).map(|_| normal.sample(&mut rng)).collect();

                    let sim_z = DMatrix::from_row_slice(n, p, &sim_data);
                    let xb = &sim_z * &sigma_half;

                    if let Ok((sim_h, _)) = royston_h_statistic(&xb) {
                        usize::from(sim_h >= h_stat)
                    } else {
                        0
                    }
                })
                .sum();

            #[cfg(not(feature = "parallel"))]
            let greater_count: usize = (0..replicates)
                .map(|_| {
                    let mut rng = thread_rng();
                    let sim_data: Vec<f64> =
                        (0..(n * p)).map(|_| normal.sample(&mut rng)).collect();

                    let sim_z = DMatrix::from_row_slice(n, p, &sim_data);
                    let xb = &sim_z * &sigma_half;

                    if let Ok((sim_h, _)) = royston_h_statistic(&xb) {
                        usize::from(sim_h >= h_stat)
                    } else {
                        0
                    }
                })
                .sum();

            (greater_count as f64 + 1.0) / (replicates as f64 + 1.0)
        },
    };

    Ok(Computation {
        statistic: T::from(h_stat).unwrap(),
        p_value: T::from(p_value).unwrap(),
    })
}
