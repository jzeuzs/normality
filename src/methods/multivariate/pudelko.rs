use nalgebra::{DMatrix, DVector, RealField, SymmetricEigen};
use num_complex::Complex;
use rand::distributions::Distribution;
use rand::thread_rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use statrs::distribution::{Normal, Uniform};

use crate::{Computation, Error, Float};

/// Performs the Pudelko test for multivariate normality.
///
/// This test is based on the empirical characteristic function. It searches for the supremum
/// of the difference between the empirical and theoretical characteristic functions within
/// a ball of radius `r`.
///
/// Takes an argument `data` which is an iterator of iterators representing the dataset (rows are
/// observations).
///
/// Also takes `r`, which is the radius of the ball for the optimization search (typically 2.0).
///
/// Finally, takes `replicates` which is the number of Monte Carlo replicates to use for the p-value
/// calculation. This is the only supported method as the asymptotic distribution is not
/// closed-form.
///
/// # Examples
///
/// ```
/// use normality::multivariate::pudelko;
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
/// // Run test with radius 2.0 and 100 Monte Carlo replicates
/// let result = pudelko(data, 2.0, 100).unwrap();
/// assert!(result.statistic > 0.5);
/// ```
pub fn pudelko<T: Float + RealField, I: IntoIterator<Item = J>, J: IntoIterator<Item = T>>(
    data: I,
    r: f64,
    replicates: usize,
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

    if n <= d {
        return Err(Error::InsufficientSampleSize {
            given: n,
            needed: d + 1,
        });
    }

    if r <= 0.0 {
        return Err(Error::Other("Radius r must be > 0".into()));
    }

    let x_mat = DMatrix::from_row_slice(n, d, &flat_data);
    let stat_obs = calculate_pudelko_statistic(&x_mat, r);
    let p_value = run_monte_carlo_p_value::<T>(n, d, stat_obs, r, replicates);

    Ok(Computation {
        statistic: T::from(stat_obs).unwrap(),
        p_value: T::from(p_value).unwrap(),
    })
}

fn calculate_pudelko_statistic<T: Float + RealField>(x_mat: &DMatrix<T>, r: f64) -> f64 {
    let n = x_mat.nrows();
    let d = x_mat.ncols();
    let standardized_data = standardize(x_mat);
    let it = if d > 3 { 1 } else { 5 };
    let mut best_value = f64::INFINITY;
    let mut rng = thread_rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    let uniform = Uniform::new(0.0, 1.0).unwrap();

    for _ in 0..it {
        let mut start_vec_data = vec![0.0; d];

        for val in &mut start_vec_data {
            *val = normal.sample(&mut rng);
        }

        let mut start_vec = DVector::from_vec(start_vec_data);
        let norm = start_vec.norm();

        if norm > 1e-9 {
            start_vec /= norm;
        }

        let u_val = uniform.sample(&mut rng);
        let scale = u_val * u_val * r;
        start_vec *= scale;

        let data_ref = &standardized_data;
        let objective = |x: &DVector<f64>| -> f64 { emp_diff(x, data_ref, r) };
        let (final_val, _) = nelder_mead(objective, &start_vec, 1000);

        if final_val < best_value {
            best_value = final_val;
        }
    }

    // Result is -min(vek) * sqrt(n)
    // best_value is negative (or min), so -best_value is positive max deviation.
    -best_value * (n as f64).sqrt()
}

/// Computes the difference between empirical and theoretical characteristic functions.
/// Returns a negative value for maximization via minimization.
fn emp_diff<T: Float + RealField>(x: &DVector<f64>, data: &DMatrix<T>, r: f64) -> f64 {
    let norm_x = x.norm();

    // Penalty outside radius r
    if norm_x > r {
        return norm_x - r;
    }

    if norm_x < 1e-15 {
        return 0.0;
    }

    let n = data.nrows();
    let mut sum_exp = Complex::new(0.0, 0.0);

    for i in 0..n {
        let row = data.row(i);
        let mut dot = 0.0;

        for (j, val) in row.iter().enumerate() {
            dot += x[j] * val.to_f64().unwrap();
        }

        sum_exp += Complex::new(0.0, dot).exp();
    }

    let emp_cf = sum_exp / (n as f64);
    let theo_cf = (-0.5 * norm_x * norm_x).exp();
    let diff = (emp_cf - Complex::new(theo_cf, 0.0)).norm();

    -(diff / norm_x)
}

/// Standardizes the data using the inverse square root of the covariance matrix.
fn standardize<T: Float + RealField>(data: &DMatrix<T>) -> DMatrix<T> {
    let n = data.nrows();
    let d = data.ncols();
    let n_f64 = n as f64;
    let mean_vec = data.row_mean();
    let mut centered = data.clone();

    for i in 0..n {
        let mut row = centered.row_mut(i);
        row -= mean_vec.clone();
    }

    let cov_raw = centered.transpose() * &centered;
    let cov = cov_raw.map(|v| v.to_f64().unwrap() / n_f64);
    let eigen = SymmetricEigen::new(cov);
    let mut inv_sqrt_diag = DMatrix::zeros(d, d);

    for i in 0..d {
        let val = eigen.eigenvalues[i];
        if val > 1e-12 {
            inv_sqrt_diag[(i, i)] = 1.0 / val.sqrt();
        } else {
            inv_sqrt_diag[(i, i)] = 0.0;
        }
    }

    let transform_mat_f64 = &eigen.eigenvectors * &inv_sqrt_diag * eigen.eigenvectors.transpose();
    let mut result = DMatrix::zeros(n, d);

    for i in 0..n {
        let row_vec_f64 = centered.row(i).map(|v| v.to_f64().unwrap());
        let transformed_row = &row_vec_f64 * &transform_mat_f64; // 1xD * DxD = 1xD

        for j in 0..d {
            result[(i, j)] = T::from(transformed_row[j]).unwrap();
        }
    }

    result
}

/// A simple Nelder-Mead optimizer for unconstrained minimization.
fn nelder_mead<F: Fn(&DVector<f64>) -> f64>(
    f: F,
    x0: &DVector<f64>,
    max_iter: usize,
) -> (f64, DVector<f64>) {
    let dim = x0.len();
    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;
    let mut simplex: Vec<(DVector<f64>, f64)> = Vec::with_capacity(dim + 1);
    let y0 = f(x0);

    simplex.push((x0.clone(), y0));

    let step = 0.05;

    for i in 0..dim {
        let mut x = x0.clone();

        if x[i].abs() < 1e-15 {
            x[i] = 0.00025;
        } else {
            x[i] *= 1.0 + step;
        }

        let y = f(&x);
        simplex.push((x, y));
    }

    for _iter in 0..max_iter {
        simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let (_x_best, f_best) = &simplex[0];
        let (x_worst, f_worst) = &simplex[dim];
        let (_, f_second_worst) = &simplex[dim - 1];
        let mut x_centroid = DVector::zeros(dim);

        for j in simplex.iter().take(dim) {
            x_centroid += &j.0;
        }

        x_centroid /= dim as f64;

        let x_r = &x_centroid + alpha * (&x_centroid - x_worst);
        let f_r = f(&x_r);

        if f_r >= *f_best && f_r < *f_second_worst {
            simplex[dim] = (x_r, f_r);
            continue;
        }

        if f_r < *f_best {
            let x_e = &x_centroid + gamma * (&x_r - &x_centroid);
            let f_e = f(&x_e);

            if f_e < f_r {
                simplex[dim] = (x_e, f_e);
            } else {
                simplex[dim] = (x_r, f_r);
            }

            continue;
        }

        if f_r >= *f_second_worst {
            if f_r < *f_worst {
                let x_oc = &x_centroid + rho * (&x_r - &x_centroid);
                let f_oc = f(&x_oc);

                if f_oc <= f_r {
                    simplex[dim] = (x_oc, f_oc);
                    continue;
                }
            } else {
                let x_ic = &x_centroid + rho * (x_worst - &x_centroid);
                let f_ic = f(&x_ic);

                if f_ic < *f_worst {
                    simplex[dim] = (x_ic, f_ic);
                    continue;
                }
            }
        }

        let x1 = simplex[0].0.clone();
        for j in simplex.iter_mut().take(dim + 1).skip(1) {
            j.0 = &x1 + sigma * (&j.0 - &x1);
            j.1 = f(&j.0);
        }
    }

    simplex.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    (simplex[0].1, simplex[0].0.clone())
}

fn run_monte_carlo_p_value<T: Float + RealField>(
    n: usize,
    d: usize,
    observed_stat: f64,
    r: f64,
    replicates: usize,
) -> T {
    let run_one_bootstrap = |_| -> i32 {
        let mut rng = thread_rng();
        let standard_normal = Normal::new(0.0, 1.0).unwrap();
        let mut boot_data_flat = vec![0.0; n * d];

        for val in &mut boot_data_flat {
            *val = standard_normal.sample(&mut rng);
        }

        let boot_mat = DMatrix::from_row_slice(n, d, &boot_data_flat);
        let stat = calculate_pudelko_statistic(&boot_mat, r);

        i32::from(stat >= observed_stat)
    };

    #[cfg(feature = "parallel")]
    let count = (0..replicates).into_par_iter().map(run_one_bootstrap).sum::<i32>();

    #[cfg(not(feature = "parallel"))]
    let count = (0..replicates).map(run_one_bootstrap).sum::<i32>();

    T::from(f64::from(count) / replicates as f64).unwrap()
}
