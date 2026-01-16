use nalgebra::{DMatrix, RealField};
use rand::distributions::Distribution;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use statrs::distribution::{ContinuousCDF, LogNormal, Normal};

use crate::{Computation, Error, Float};

/// Specifies the method for p-value calculation in the Henze-Wagner test.
#[derive(Debug, Clone, Copy)]
pub enum HenzeWagnerMethod {
    /// Use the log-normal approximation for the p-value.
    /// This is the standard method derived by Henze and Wagner (1997).
    LogNormal,

    /// Use a parametric bootstrap (Monte Carlo) simulation with a given number of replicates.
    MonteCarlo(usize),
}

/// Performs the Henze-Wagner test for multivariate normality.
///
/// This test is a high-dimensional version of the BHEP test. It is particularly
/// useful when the dimension `d` is large relative to the sample size `n`, or
/// when the covariance matrix is singular.
///
/// It uses a Moore-Penrose pseudoinverse if the covariance matrix is singular.
///
/// Takes an argument `data` which is an iterator of iterators.
///
/// # Examples
///
/// ```
/// use normality::multivariate::{HenzeWagnerMethod, henze_wagner};
///
/// // 3D data
/// let data = vec![
///     vec![0.1, 0.2, 0.3],
///     vec![0.5, 0.1, 0.4],
///     vec![-0.2, 0.3, 0.1],
///     vec![0.0, 0.0, 0.0],
///     vec![0.8, -0.5, 0.2],
/// ];
///
/// // Use LogNormal approximation
/// let result = henze_wagner(data, false, HenzeWagnerMethod::LogNormal).unwrap();
/// assert!(result.p_value > 0.0);
/// ```
pub fn henze_wagner<T: Float + RealField, I: IntoIterator<Item = J>, J: IntoIterator<Item = T>>(
    data: I,
    use_population_covariance: bool,
    method: HenzeWagnerMethod,
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

    if n < 2 {
        return Err(Error::InsufficientSampleSize {
            given: n,
            needed: 2,
        });
    }

    let x_mat = DMatrix::from_row_slice(n, d, &flat_data);
    let hw_stat_val = calculate_hw_statistic(&x_mat, use_population_covariance)?;
    let hw_stat = T::from(hw_stat_val).unwrap();
    let p_value_f64 = match method {
        HenzeWagnerMethod::LogNormal => calculate_log_normal_p_value(hw_stat_val, d),
        HenzeWagnerMethod::MonteCarlo(replicates) => {
            run_monte_carlo_p_value::<T>(n, d, hw_stat_val, use_population_covariance, replicates)
        },
    };

    Ok(Computation {
        statistic: hw_stat,
        p_value: T::from(p_value_f64).unwrap(),
    })
}

fn calculate_hw_statistic<T: Float + RealField>(
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
        s_raw.map(|v| v / n_t)
    } else {
        s_raw.map(|v| v / T::from(n - 1).unwrap())
    };

    let s_inv = if let Some(inv) = s_mat.clone().try_inverse() {
        inv
    } else {
        let svd = s_mat.svd(true, true);
        svd.pseudo_inverse(T::from(1e-15).unwrap())
            .map_err(|_| Error::Other("Failed to compute pseudoinverse".into()))?
    };

    let x_s_inv = &x_centered * &s_inv;

    #[cfg(feature = "parallel")]
    let d_diag: Vec<T> =
        (0..n).into_par_iter().map(|i| x_centered.row(i).dot(&x_s_inv.row(i))).collect();

    #[cfg(not(feature = "parallel"))]
    let d_diag: Vec<T> = (0..n).map(|i| x_centered.row(i).dot(&x_s_inv.row(i))).collect();

    let n_f64 = n as f64;
    let d_f64 = d as f64;
    let b: f64 = 1.0;
    let b_sq = b * b;
    let big_d_matrix = &x_centered * x_s_inv.transpose();

    #[cfg(feature = "parallel")]
    let sum_exp_djk: f64 = (0..n)
        .into_par_iter()
        .map(|j| {
            let dj = d_diag[j].to_f64().unwrap();
            let mut local_sum = 0.0;

            for k in 0..n {
                let dk = d_diag[k].to_f64().unwrap();
                let d_cross = big_d_matrix[(j, k)].to_f64().unwrap();
                let d_jk = dj + dk - 2.0 * d_cross;

                local_sum += (-b_sq / 2.0 * d_jk).exp();
            }

            local_sum
        })
        .sum();

    #[cfg(not(feature = "parallel"))]
    let sum_exp_djk: f64 = (0..n)
        .map(|j| {
            let dj = d_diag[j].to_f64().unwrap();
            let mut local_sum = 0.0;

            for k in 0..n {
                let dk = d_diag[k].to_f64().unwrap();
                let d_cross = big_d_matrix[(j, k)].to_f64().unwrap();
                let d_jk = dj + dk - 2.0 * d_cross;

                local_sum += (-b_sq / 2.0 * d_jk).exp();
            }

            local_sum
        })
        .sum();

    let part1 = sum_exp_djk / (n_f64 * n_f64);
    let sum_exp_dj: f64 =
        d_diag.iter().map(|val| (-b_sq / (2.0 * (1.0 + b_sq)) * val.to_f64().unwrap()).exp()).sum();

    let part2 = 2.0 * (1.0 + b_sq).powf(-d_f64 / 2.0) * sum_exp_dj / n_f64;
    let part3 = (1.0 + 2.0 * b_sq).powf(-d_f64 / 2.0);
    let hw = n_f64 * (part1 - part2 + part3);

    Ok(hw)
}

fn calculate_log_normal_p_value(hw: f64, d: usize) -> f64 {
    let p = d as f64;
    let b: f64 = 1.0;
    let b2 = b * b;
    let b4 = b2 * b2;
    let b8 = b4 * b4;
    let a = 1.0 + 2.0 * b2;
    let wb = (1.0 + b2) * (1.0 + 3.0 * b2);
    let mu = 1.0 - a.powf(-p / 2.0) * (1.0 + (p * b2) / a + (p * (p + 2.0) * b4) / (2.0 * a * a));
    let term1 = 2.0 * (1.0 + 4.0 * b2).powf(-p / 2.0);
    let term2 = 2.0
        * a.powf(-p)
        * (1.0 + (2.0 * p * b4) / (a * a) + (3.0 * p * (p + 2.0) * b8) / (4.0 * a.powf(4.0)));

    let term3 = 4.0
        * wb.powf(-p / 2.0)
        * (1.0 + (3.0 * p * b4) / (2.0 * wb) + (p * (p + 2.0) * b8) / (2.0 * wb * wb));

    let si2 = term1 + term2 - term3;
    let mu_sq = mu * mu;
    let pmu = (mu_sq * mu_sq / (si2 + mu_sq)).sqrt().ln();
    let psi = ((si2 + mu_sq) / mu_sq).ln().sqrt();
    let dist = LogNormal::new(pmu, psi).unwrap();

    dist.sf(hw)
}

fn run_monte_carlo_p_value<T: Float + RealField>(
    n: usize,
    d: usize,
    observed_stat: f64,
    use_population_covariance: bool,
    replicates: usize,
) -> f64 {
    // Note: Henze-Wagner is affine invariant. We can sample from Standard Normal
    // rather than N(mu, Sigma) to save computation time on decomposing singular matrices
    // inside the loop, while maintaining mathematical equivalence for the test statistic
    // distribution.

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

            match calculate_hw_statistic(&boot_mat, use_population_covariance) {
                Ok(stat) => (i32::from(stat >= observed_stat), 1),
                Err(_) => (0, 0),
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

            if let Ok(stat) = calculate_hw_statistic(&boot_mat, use_population_covariance) {
                valid_replicates += 1;
                if stat >= observed_stat {
                    count += 1;
                }
            }
        }
        (count, valid_replicates)
    };

    if valid_replicates > 0 { f64::from(count) / f64::from(valid_replicates) } else { f64::NAN }
}
