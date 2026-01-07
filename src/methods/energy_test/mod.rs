mod imhof;

use std::f64::consts::PI;
use std::iter::Sum;

use rand::distributions::Distribution;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use statrs::distribution::{Continuous, ContinuousCDF, Normal};

use crate::{Computation, Error, Float};

/// Specifies the method for p-value calculation in the Energy test.
#[derive(Debug, Clone, Copy)]
pub enum EnergyTestMethod {
    /// Use a Monte Carlo (parametric bootstrap) simulation with a given number of replicates.
    /// A larger number (e.g., 1000+) yields a more stable p-value.
    MonteCarlo(usize),
    /// Use the asymptotic limit distribution. The p-value is calculated
    /// numerically using the Imhof (1961) algorithm and pre-computed eigenvalues
    /// for the composite hypothesis (Case 4).
    Asymptotic,
}

/// Performs the Energy test for univariate normality.
///
/// It assesses the null hypothesis that the data comes from a normal distribution
/// with unknown mean and variance (Case 4).
///
/// It calculates the Energy statistic (`E`) and determines the p-value using
/// either a parametric bootstrap (Monte Carlo) or the asymptotic distribution.
///
/// # Arguments
///
/// * `data` - An iterator over floating-point numbers (`impl IntoIterator<Item = T>`). Must contain
///   at least 5 elements.
/// * `method` - The method for calculating the p-value
///   ([`MonteCarlo`](EnergyTestMethod::MonteCarlo) or
///   [`Asymptotic`](EnergyTestMethod::Asymptotic)).
///
/// # Examples
///
/// ```
/// use normality::{Computation, EnergyTestMethod, energy_test};
///
/// let normal_data = vec![-0.9, -0.6, -0.3, 0.0, 0.2, 0.5, 0.8, 1.1, 1.4, 1.6];
///
/// // Use the Asymptotic method (fast, deterministic)
/// let result = energy_test(normal_data.iter().copied(), EnergyTestMethod::Asymptotic).unwrap();
/// assert!(result.p_value > 0.05);
///
/// // Use the Monte Carlo method (robust)
/// let mc_result = energy_test(normal_data, EnergyTestMethod::MonteCarlo(1000)).unwrap();
/// assert!(mc_result.p_value > 0.05);
/// ```
pub fn energy_test<T: Float, I: IntoIterator<Item = T>>(
    data: I,
    method: EnergyTestMethod,
) -> Result<Computation<T>, Error> {
    let mut data_vec: Vec<T> = data.into_iter().collect();
    let n = data_vec.len();

    if n < 5 {
        return Err(Error::InsufficientSampleSize {
            needed: 5,
            given: n,
        });
    }

    // Calculate the observed statistic from the original data
    let observed_statistic = calculate_energy_statistic_internal(&mut data_vec)?;
    let statistic_f64 = observed_statistic.to_f64().unwrap();

    let p_value_f64 = match method {
        EnergyTestMethod::MonteCarlo(replicates) => {
            if replicates == 0 {
                f64::NAN
            } else {
                run_monte_carlo_p_value(n, observed_statistic, replicates)
            }
        },
        EnergyTestMethod::Asymptotic => {
            // Eigenvalues for Case 4 (Composite Hypothesis), d=1.
            // Source: EVnormal data from R 'energy' package / Móri et al. (2021)
            #[allow(clippy::unreadable_literal)]
            const EIGENVALUES: [f64; 125] = [
                1.131075e-01,
                8.356687e-02,
                3.911317e-02,
                3.182242e-02,
                1.990113e-02,
                1.697827e-02,
                1.206876e-02,
                1.059855e-02,
                8.104870e-03,
                7.258995e-03,
                5.820330e-03,
                5.288066e-03,
                4.383281e-03,
                4.026155e-03,
                3.420411e-03,
                3.168924e-03,
                2.743654e-03,
                2.559740e-03,
                2.249808e-03,
                2.111158e-03,
                1.878287e-03,
                1.770952e-03,
                1.590384e-03,
                1.503884e-03,
                1.354607e-03,
                1.278480e-03,
                1.142998e-03,
                1.071186e-03,
                9.450468e-04,
                8.781150e-04,
                7.644091e-04,
                7.043642e-04,
                6.057134e-04,
                5.536943e-04,
                4.708477e-04,
                4.270931e-04,
                3.594278e-04,
                3.235633e-04,
                2.696432e-04,
                2.409256e-04,
                1.989138e-04,
                1.764121e-04,
                1.443566e-04,
                1.270824e-04,
                1.031023e-04,
                9.009718e-05,
                7.249299e-05,
                6.288386e-05,
                5.019235e-05,
                4.322003e-05,
                3.422906e-05,
                2.925839e-05,
                2.299633e-05,
                1.951306e-05,
                1.522330e-05,
                1.282303e-05,
                9.931578e-06,
                8.304619e-06,
                6.386348e-06,
                5.301256e-06,
                4.048294e-06,
                3.336021e-06,
                2.530065e-06,
                2.069788e-06,
                1.559132e-06,
                1.266257e-06,
                9.474862e-07,
                7.639510e-07,
                5.678669e-07,
                4.545711e-07,
                3.356956e-07,
                2.667924e-07,
                1.957537e-07,
                1.544618e-07,
                1.126100e-07,
                8.822313e-08,
                6.391205e-08,
                4.971580e-08,
                3.579000e-08,
                2.764343e-08,
                1.977639e-08,
                1.516730e-08,
                1.078377e-08,
                8.212517e-09,
                5.803148e-09,
                4.388608e-09,
                3.082164e-09,
                2.314676e-09,
                1.615758e-09,
                1.205024e-09,
                8.360911e-10,
                6.192589e-10,
                4.270861e-10,
                3.141576e-10,
                2.153719e-10,
                1.573435e-10,
                1.072261e-10,
                7.780409e-11,
                5.270801e-11,
                3.798698e-11,
                2.558246e-11,
                1.831350e-11,
                1.226091e-11,
                8.718405e-12,
                5.802868e-12,
                4.098800e-12,
                2.712230e-12,
                1.903069e-12,
                1.251981e-12,
                8.726784e-13,
                5.707954e-13,
                3.952568e-13,
                2.570382e-13,
                1.768288e-13,
                1.143332e-14,
                7.814448e-14,
                5.023732e-14,
                3.411382e-14,
                2.180656e-14,
                1.471254e-14,
                9.350826e-15,
                6.268849e-15,
                6.221851e-15,
                3.962057e-15,
                3.662914e-15,
            ];

            imhof::significance_level(statistic_f64, &EIGENVALUES)?
        },
    };

    Ok(Computation {
        statistic: observed_statistic,
        p_value: T::from(p_value_f64).unwrap(),
    })
}

fn calculate_energy_statistic_internal<T: Float>(data: &mut [T]) -> Result<T, Error> {
    if data.iter().any(|v| v.is_nan()) {
        return Err(Error::ContainsNaN);
    }

    let n = data.len();
    let n_t = T::from(n).unwrap();
    let sum: T = iter_if_parallel!(data).copied().sum();
    let mean = sum / n_t;
    let variance: T =
        iter_if_parallel!(data).map(|&x| (x - mean).powi(2)).sum::<T>() / T::from(n - 1).unwrap();

    let std_dev = variance.sqrt();

    if std_dev <= T::zero() {
        return Err(Error::ZeroRange);
    }

    #[cfg(feature = "parallel")]
    data.par_iter_mut().for_each(|x| *x = (*x - mean) / std_dev);

    #[cfg(not(feature = "parallel"))]
    data.iter_mut().for_each(|x| *x = (*x - mean) / std_dev);

    sort_if_parallel!(data, |a, b| a.partial_cmp(b).unwrap());

    let standard_normal = Normal::new(0.0, 1.0).unwrap();

    #[cfg(feature = "parallel")]
    let (sum_term_1, sum_term_k) = data
        .par_iter()
        .enumerate()
        .map(|(i, &y)| {
            let y_f64 = y.to_f64().unwrap();
            let phi = T::from(standard_normal.cdf(y_f64)).unwrap();
            let pdf = T::from(standard_normal.pdf(y_f64)).unwrap();

            let term1 = (T::from(2.0).unwrap() * y * phi) + (T::from(2.0).unwrap() * pdf);

            // K_i = (1 - n) + 2 * i (0-indexed)
            let k_i = T::from(1.0 - (n as f64) + 2.0 * (i as f64)).unwrap();
            let termk = k_i * y;
            (term1, termk)
        })
        .reduce(|| (T::zero(), T::zero()), |a, b| (a.0 + b.0, a.1 + b.1));

    #[cfg(not(feature = "parallel"))]
    let (sum_term_1, sum_term_k) =
        data.iter().enumerate().fold((T::zero(), T::zero()), |acc, (i, &y)| {
            let y_f64 = y.to_f64().unwrap();
            let phi = T::from(standard_normal.cdf(y_f64)).unwrap();
            let pdf = T::from(standard_normal.pdf(y_f64)).unwrap();

            let term1 = (T::from(2.0).unwrap() * y * phi) + (T::from(2.0).unwrap() * pdf);
            let k_i = T::from(1.0 - (n as f64) + 2.0 * (i as f64)).unwrap();
            let termk = k_i * y;
            (acc.0 + term1, acc.1 + termk)
        });

    let sqrt_pi = T::from(PI.sqrt()).unwrap();
    let term_2 = n_t / sqrt_pi;
    let term_3 = sum_term_k / n_t; // mean(K * y)

    let statistic = T::from(2.0).unwrap() * (sum_term_1 - term_2 - term_3);

    Ok(statistic)
}

fn run_monte_carlo_p_value<T>(n: usize, observed_stat: T, replicates: usize) -> f64
where
    T: Float + Sum,
{
    #[allow(unused_assignments)]
    let mut count = 0;

    #[cfg(feature = "parallel")]
    {
        count = (0..replicates)
            .into_par_iter()
            .map(|_| {
                let mut rng = rand::thread_rng(); // Use thread-local RNG
                let normal_dist = Normal::new(0.0, 1.0).unwrap();
                let mut buffer = vec![T::zero(); n];
                for x in &mut buffer {
                    *x = T::from(normal_dist.sample(&mut rng)).unwrap();
                }

                let sim_stat = calculate_energy_statistic_internal(&mut buffer).unwrap();
                i32::from(sim_stat >= observed_stat)
            })
            .sum();
    }

    #[cfg(not(feature = "parallel"))]
    {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::from_entropy();
        let normal_dist = Normal::new(0.0, 1.0).unwrap();
        let mut buffer = vec![T::zero(); n];

        for _ in 0..replicates {
            for x in &mut buffer {
                *x = T::from(normal_dist.sample(&mut rng)).unwrap();
            }

            let sim_stat = calculate_energy_statistic_internal(&mut buffer).unwrap();
            if sim_stat >= observed_stat {
                count += 1;
            }
        }
    }

    (f64::from(count) + 1.0) / (replicates as f64 + 1.0)
}
