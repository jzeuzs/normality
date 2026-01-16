use std::f64::consts::SQRT_2;

use nalgebra::{DMatrix, RealField};
use rand::distributions::Distribution;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, Normal};

use crate::{Computation, Error, Float};

/// Specifies the method for p-value calculation in the Henze-Zirkler test.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum HenzeZirklerMethod {
    /// Use the log-normal approximation for the p-value.
    /// This is the standard method derived by Henze and Zirkler (1990).
    LogNormal,

    /// Use a parametric bootstrap (Monte Carlo) simulation with a given number of replicates.
    MonteCarlo(usize),
}

/// Performs the Henze-Zirkler test for multivariate normality.
///
/// The test is an affine-invariant, consistent test for checking if a dataset of
/// d-dimensional vectors comes from a multivariate normal distribution.
///
/// It calculates the HZ statistic based on a weighted L^2-distance between the empirical
/// characteristic function and the theoretical normal characteristic function.
///
///
/// Takes an argument `data` which is an iterator of iterators (`impl IntoIterator<Item = impl
/// IntoIterator<Item = T>>`). Each inner iterator represents an observation of dimension d. All
/// observations must have the same dimension. The sample size n must be greater than d + 1.
///
/// Also takes `use_population_covariance` which, if `true`, the covariance matrix is estimated
/// using the population formula (1/N). If `false`, the sample formula (1/(n-1)) is used.
///
/// Lastly, `method` which is the method for calculating the p-value
/// ([`LogNormal`](HenzeZirklerMethod::LogNormal) or
/// [`MonteCarlo`](HenzeZirklerMethod::MonteCarlo)).
///
/// # Examples
///
/// ```
/// use normality::Computation;
/// use normality::multivariate::{HenzeZirklerMethod, henze_zirkler};
///
/// // 3D data from a multivariate normal distribution
/// let normal_data = vec![
///     vec![0.1, 0.2, 0.3],
///     vec![0.5, 0.1, 0.4],
///     vec![-0.2, 0.3, 0.1],
///     vec![0.0, 0.0, 0.0],
///     vec![0.8, -0.5, 0.2],
///     vec![-0.1, -0.1, -0.1],
/// ];
///
/// // Use LogNormal approximation with population covariance (standard)
/// let result = henze_zirkler(normal_data.clone(), false, HenzeZirklerMethod::LogNormal).unwrap();
/// assert!(result.p_value > 0.05);
///
/// // Use Monte Carlo simulation with 1000 replicates
/// let mc_result =
///     henze_zirkler(normal_data, false, HenzeZirklerMethod::MonteCarlo(1000)).unwrap();
/// assert!(mc_result.p_value > 0.05);
/// ```
pub fn henze_zirkler<T: Float + RealField, I: IntoIterator<Item = J>, J: IntoIterator<Item = T>>(
    data: I,
    use_population_covariance: bool,
    method: HenzeZirklerMethod,
) -> Result<Computation<T>, Error> {
    let mut flat_data = Vec::new();
    let mut n = 0;
    let mut d = 0;

    for (i, row) in data.into_iter().enumerate() {
        n += 1;
        let mut row_len = 0;

        for val in row {
            if val.is_nan() {
                return Err(Error::ContainsNaN);
            }
            flat_data.push(val);
            row_len += 1;
        }

        if i == 0 {
            d = row_len;

            if d == 0 {
                return Err(Error::DimensionMismatch);
            }
        } else if row_len != d {
            return Err(Error::DimensionMismatch);
        }
    }

    if n == 0 {
        return Err(Error::InsufficientSampleSize {
            given: 0,
            needed: d + 1,
        });
    }

    if d < 2 {
        return Err(Error::InsufficientSampleSize {
            given: n,
            needed: d + 2,
        });
    }

    let x_mat = DMatrix::from_row_slice(n, d, &flat_data);
    let hz_stat_val = calculate_hz_statistic(&x_mat, use_population_covariance)?;
    let hz_stat = T::from(hz_stat_val).unwrap();

    let p_value_f64 = match method {
        HenzeZirklerMethod::LogNormal => calculate_log_normal_p_value(hz_stat_val, n, d),
        HenzeZirklerMethod::MonteCarlo(replicates) => {
            run_monte_carlo_p_value::<T>(n, d, hz_stat_val, use_population_covariance, replicates)
        },
    };

    Ok(Computation {
        statistic: hz_stat,
        p_value: T::from(p_value_f64).unwrap(),
    })
}

fn calculate_hz_statistic<T: Float + RealField>(
    x_mat: &DMatrix<T>,
    use_population_covariance: bool,
) -> Result<f64, Error> {
    let n = x_mat.nrows();
    let d = x_mat.ncols();
    let n_t = T::from(n).unwrap();
    let mean_vec = x_mat.row_mean().transpose();
    let mut x_centered = x_mat.clone();

    for i in 0..n {
        let mut row = x_centered.row_mut(i);
        row -= mean_vec.transpose();
    }

    let s_raw = x_centered.transpose() * &x_centered;
    let s_mat = if use_population_covariance {
        s_raw.map(|v| v / n_t) // 1/n (Population)
    } else {
        s_raw.map(|v| v / T::from(n - 1).unwrap()) // 1/(n-1) (Sample)
    };

    let s_inv = if let Some(inv) = s_mat.clone().try_inverse() {
        inv
    } else {
        let svd = s_mat.svd(true, true);
        svd.pseudo_inverse(T::from(1e-15).unwrap())
            .map_err(|_| Error::Other("Failed to compute pseudoinverse".into()))?
    };

    let x_s_inv = &x_centered * &s_inv; // (n x d) * (d x d) = (n x d)

    #[cfg(feature = "parallel")]
    let d_sq: Vec<T> =
        (0..n).into_par_iter().map(|i| x_centered.row(i).dot(&x_s_inv.row(i))).collect();

    #[cfg(not(feature = "parallel"))]
    let d_sq: Vec<T> = (0..n).map(|i| x_centered.row(i).dot(&x_s_inv.row(i))).collect();

    let n_f64 = n as f64;
    let d_f64 = d as f64;
    let exponent = 1.0 / (d_f64 + 4.0);
    let b = (1.0 / SQRT_2) * ((2.0 * d_f64 + 1.0) / 4.0).powf(exponent) * n_f64.powf(exponent);
    let b_sq = b * b;
    let big_d_matrix = &x_centered * x_s_inv.transpose(); // (n x n) matrix of D_ij

    #[cfg(feature = "parallel")]
    let sum_exp_djk: f64 = (0..n)
        .into_par_iter()
        .map(|i| {
            let di = d_sq[i].to_f64().unwrap();
            let mut local_sum = 0.0;

            for j in 0..n {
                let dj = d_sq[j].to_f64().unwrap();
                let dij = big_d_matrix[(i, j)].to_f64().unwrap();
                let dist_sq = di + dj - 2.0 * dij;

                local_sum += (-b_sq / 2.0 * dist_sq).exp();
            }

            local_sum
        })
        .sum();

    #[cfg(not(feature = "parallel"))]
    let sum_exp_djk: f64 = (0..n)
        .map(|i| {
            let di = d_sq[i].to_f64().unwrap();
            let mut local_sum = 0.0;

            for j in 0..n {
                let dj = d_sq[j].to_f64().unwrap();
                let dij = big_d_matrix[(i, j)].to_f64().unwrap();
                let dist_sq = di + dj - 2.0 * dij;

                local_sum += (-b_sq / 2.0 * dist_sq).exp();
            }

            local_sum
        })
        .sum();

    let part1 = sum_exp_djk / (n_f64 * n_f64);
    let sum_exp_dj: f64 =
        d_sq.iter().map(|val| (-b_sq / (2.0 * (1.0 + b_sq)) * val.to_f64().unwrap()).exp()).sum();

    let part2 = 2.0 * (1.0 + b_sq).powf(-d_f64 / 2.0) * sum_exp_dj / n_f64;
    let part3 = (1.0 + 2.0 * b_sq).powf(-d_f64 / 2.0);
    let hz = n_f64 * (part1 - part2 + part3);

    Ok(hz)
}

fn calculate_log_normal_p_value(hz: f64, n: usize, d: usize) -> f64 {
    let d_f64 = d as f64;
    let n_f64 = n as f64;
    let exponent = 1.0 / (d_f64 + 4.0);
    let b = (1.0 / SQRT_2) * ((2.0 * d_f64 + 1.0) / 4.0).powf(exponent) * n_f64.powf(exponent);
    let b2 = b * b;
    let b4 = b2 * b2;
    let b8 = b4 * b4;
    let a = 1.0 + 2.0 * b2;
    let wb = (1.0 + b2) * (1.0 + 3.0 * b2);
    let mu = 1.0
        - a.powf(-d_f64 / 2.0)
            * (1.0 + (d_f64 * b2) / a + (d_f64 * (d_f64 + 2.0) * b4) / (2.0 * a * a));

    let si2 = 2.0 * (1.0 + 4.0 * b2).powf(-d_f64 / 2.0)
        + 2.0
            * a.powf(-d_f64)
            * (1.0
                + (2.0 * d_f64 * b4) / (a * a)
                + (3.0 * d_f64 * (d_f64 + 2.0) * b8) / (4.0 * a.powi(4)))
        - 4.0
            * wb.powf(-d_f64 / 2.0)
            * (1.0
                + (3.0 * d_f64 * b4) / (2.0 * wb)
                + (d_f64 * (d_f64 + 2.0) * b8) / (2.0 * wb * wb));

    let mu_sq = mu * mu;
    let pmu = (mu_sq * mu_sq / (si2 + mu_sq)).sqrt().ln();
    let psi = ((si2 + mu_sq) / mu_sq).ln().sqrt();
    let dist_z = (hz.ln() - pmu) / psi;
    let standard_normal_dist = Normal::new(0.0, 1.0).unwrap();

    standard_normal_dist.sf(dist_z)
}

fn run_monte_carlo_p_value<T: Float + RealField>(
    n: usize,
    d: usize,
    observed_hz: f64,
    use_population_covariance: bool,
    replicates: usize,
) -> f64 {
    #[cfg(feature = "parallel")]
    let (count, valid_replicates) = (0..replicates)
        .into_par_iter()
        .map(|_| {
            let mut rng = rand::thread_rng();
            let standard_normal = Normal::new(0.0, 1.0).unwrap();
            let mut boot_data_flat = vec![T::zero(); n * d];

            for val in &mut boot_data_flat {
                *val = T::from(standard_normal.sample(&mut rng)).unwrap();
            }

            let boot_mat = DMatrix::from_row_slice(n, d, &boot_data_flat);

            match calculate_hz_statistic(&boot_mat, use_population_covariance) {
                Ok(hz_val) => (i32::from(hz_val >= observed_hz), 1),
                Err(_) => (0, 0), // Skip singular
            }
        })
        .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1));

    #[cfg(not(feature = "parallel"))]
    let (count, valid_replicates) = {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::from_entropy();
        let standard_normal = Normal::new(0.0, 1.0).unwrap();
        let mut count = 0;
        let mut valid_replicates = 0;
        let mut boot_data_flat = vec![T::zero(); n * d];

        for _ in 0..replicates {
            for val in &mut boot_data_flat {
                *val = T::from(standard_normal.sample(&mut rng)).unwrap();
            }

            let boot_mat = DMatrix::from_row_slice(n, d, &boot_data_flat);

            if let Ok(hz_val) = calculate_hz_statistic(&boot_mat, use_population_covariance) {
                valid_replicates += 1;
                if hz_val >= observed_hz {
                    count += 1;
                }
            }
        }

        (count, valid_replicates)
    };

    if valid_replicates > 0 { f64::from(count) / f64::from(valid_replicates) } else { f64::NAN }
}
