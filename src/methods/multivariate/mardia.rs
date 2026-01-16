use nalgebra::{ComplexField, DMatrix, RealField, SymmetricEigen};
use rand::distributions::Distribution;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

use crate::{Computation, Error, Float};

/// Specifies the method for p-value calculation in Mardia's test.
#[derive(Debug, Clone, Copy)]
pub enum MardiaMethod {
    /// Use the asymptotic distributions (Chi-Squared for skewness, Normal for kurtosis).
    Asymptotic,

    /// Use a parametric bootstrap (Monte Carlo) simulation with a given number of replicates.
    MonteCarlo(usize),
}

/// Holds the results for both of Mardia's tests (Skewness and Kurtosis).
#[derive(Debug, PartialEq)]
pub struct MardiaComputation<T: Float> {
    /// The result of Mardia's Multivariate Skewness test.
    pub skewness: Computation<T>,

    /// The result of Mardia's Multivariate Kurtosis test.
    pub kurtosis: Computation<T>,
}

/// Performs Mardia’s skewness and kurtosis tests to assess multivariate normality.
///
/// This function calculates Mahalanobis distances to compute multivariate skewness and kurtosis.
/// It includes a small-sample correction for the skewness statistic when `n < 20`.
///
/// Takes an argument `data` which is an iterator of iterators representing the dataset (rows are
/// observations).
///
/// Also takes `use_population_covariance`, which, if `true`, uses the population covariance
/// estimator (n divisor); otherwise uses sample covariance (n-1 divisor).
///
/// Lastly, takes `method` which is the method to calculate p-values
/// ([`Asymptotic`](MardiaMethod::Asymptotic) or [`MonteCarlo`](MardiaMethod::MonteCarlo)).
///
/// # Examples
///
/// ```
/// use normality::multivariate::{MardiaMethod, mardia};
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
/// // Use Asymptotic approximation
/// let result = mardia(data, false, MardiaMethod::Asymptotic).unwrap();
/// assert!(result.skewness.p_value > 0.05);
/// assert!(result.kurtosis.p_value > 0.05);
/// ```
pub fn mardia<T: Float + RealField, I: IntoIterator<Item = J>, J: IntoIterator<Item = T>>(
    data: I,
    use_population_covariance: bool,
    method: MardiaMethod,
) -> Result<MardiaComputation<T>, Error> {
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
    let (skew_stat_obs, kurt_stat_obs) =
        calculate_mardia_statistics(&x_mat, use_population_covariance)?;

    let (p_skew, p_kurt) = match method {
        MardiaMethod::Asymptotic => {
            calculate_asymptotic_p_values(n, d, skew_stat_obs, kurt_stat_obs)
        },
        MardiaMethod::MonteCarlo(replicates) => run_monte_carlo_p_values::<T>(
            n,
            d,
            &x_mat,
            skew_stat_obs,
            kurt_stat_obs,
            use_population_covariance,
            replicates,
        ),
    };

    Ok(MardiaComputation {
        skewness: Computation {
            statistic: T::from(skew_stat_obs).unwrap(),
            p_value: T::from(p_skew).unwrap(),
        },
        kurtosis: Computation {
            statistic: T::from(kurt_stat_obs).unwrap(),
            p_value: T::from(p_kurt).unwrap(),
        },
    })
}

fn calculate_mardia_statistics<T: Float + RealField>(
    x_mat: &DMatrix<T>,
    use_population_covariance: bool,
) -> Result<(f64, f64), Error> {
    let n = x_mat.nrows();
    let p = x_mat.ncols();
    let n_f64 = n as f64;
    let p_f64 = p as f64;
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

    let d_mat = &x_centered * &s_inv * x_centered.transpose();
    let sum_d_cubed: f64 = d_mat.iter().map(|&v| v.to_f64().unwrap().powi(3)).sum();
    let g1p = sum_d_cubed / (n_f64 * n_f64);
    let sum_diag_sq: f64 = d_mat.diagonal().iter().map(|&v| v.to_f64().unwrap().powi(2)).sum();
    let g2p = sum_diag_sq / n_f64;
    let k_const = ((p_f64 + 1.0) * (n_f64 + 1.0) * (n_f64 + 3.0))
        / (n_f64 * ((n_f64 + 1.0) * (p_f64 + 1.0) - 6.0));

    let skew_stat = if n < 20 { n_f64 * k_const * g1p / 6.0 } else { n_f64 * g1p / 6.0 };
    let expected_kurt = p_f64 * (p_f64 + 2.0);
    let variance_kurt = 8.0 * p_f64 * (p_f64 + 2.0) / n_f64;
    let kurt_stat = (g2p - expected_kurt) / variance_kurt.sqrt();

    Ok((skew_stat, kurt_stat))
}

fn calculate_asymptotic_p_values(
    _n: usize,
    d: usize,
    skew_stat: f64,
    kurt_stat: f64,
) -> (f64, f64) {
    let p = d as f64;
    let df_skew = p * (p + 1.0) * (p + 2.0) / 6.0;
    let dist_skew = ChiSquared::new(df_skew).unwrap();
    let p_skew = dist_skew.sf(skew_stat); // Upper tail
    let dist_kurt = Normal::new(0.0, 1.0).unwrap();
    let p_kurt = 2.0 * dist_kurt.sf(kurt_stat.abs());

    (p_skew, p_kurt)
}

fn run_monte_carlo_p_values<T: Float + RealField>(
    n: usize,
    d: usize,
    orig_x: &DMatrix<T>,
    obs_skew: f64,
    obs_kurt: f64,
    use_pop_cov: bool,
    replicates: usize,
) -> (f64, f64) {
    let n_t = T::from(n).unwrap();
    let mean_vec = orig_x.row_mean();
    let mut x_centered = orig_x.clone();

    for i in 0..n {
        let mut row = x_centered.row_mut(i);
        row -= mean_vec.transpose();
    }

    let s_raw = x_centered.transpose() * &x_centered;
    let sigma_hat = if use_pop_cov {
        s_raw.map(|v| v / n_t)
    } else {
        s_raw.map(|v| v / T::from(n - 1).unwrap())
    };

    let eigen = SymmetricEigen::new(sigma_hat.clone());
    let eigen_vecs = eigen.eigenvectors;
    let eigen_vals = eigen.eigenvalues;
    let mut sqrt_eigen_vals = DMatrix::zeros(d, d);

    for i in 0..d {
        let val = if eigen_vals[i] < T::zero() { T::zero() } else { eigen_vals[i] };
        sqrt_eigen_vals[(i, i)] = ComplexField::sqrt(val);
    }

    let transform_mat = &eigen_vecs * &sqrt_eigen_vals; // (d x d)
    let run_one_bootstrap = |_| -> (i32, i32, i32) {
        let mut rng = rand::thread_rng();
        let standard_normal = Normal::new(0.0, 1.0).unwrap();
        let mut z_data = vec![T::zero(); n * d];

        for val in &mut z_data {
            *val = T::from(standard_normal.sample(&mut rng)).unwrap();
        }

        let z_mat = DMatrix::from_row_slice(n, d, &z_data);
        let mut x_boot = &z_mat * transform_mat.transpose();

        for i in 0..n {
            let mut row = x_boot.row_mut(i);
            row += mean_vec.transpose();
        }

        match calculate_mardia_statistics(&x_boot, use_pop_cov) {
            Ok((s, k)) => {
                let s_hit = i32::from(s >= obs_skew);
                let k_hit = i32::from(k.abs() >= obs_kurt.abs());
                (s_hit, k_hit, 1)
            },
            Err(_) => (0, 0, 0),
        }
    };

    #[cfg(feature = "parallel")]
    let (count_skew, count_kurt, valid_reps) = (0..replicates)
        .into_par_iter()
        .map(run_one_bootstrap)
        .reduce(|| (0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

    #[cfg(not(feature = "parallel"))]
    let (count_skew, count_kurt, valid_reps) = (0..replicates)
        .map(run_one_bootstrap)
        .fold((0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

    if valid_reps == 0 {
        return (f64::NAN, f64::NAN);
    }

    let p_skew = f64::from(count_skew) / f64::from(valid_reps);
    let p_kurt = f64::from(count_kurt) / f64::from(valid_reps);

    (p_skew, p_kurt)
}
