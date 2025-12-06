use std::f64::consts::PI;

use eqsolver::integrators::NewtonCotes;

struct ImhofParams<'a> {
    q: f64,
    lambda: &'a [f64],
}

/// The theta function from Imhof (1961), Eq 3.2.
fn theta(u: f64, params: &ImhofParams) -> f64 {
    let mut sum_atan = 0.0;
    for &lam in params.lambda {
        sum_atan += (lam * u).atan();
    }

    0.5 * sum_atan - 0.5 * params.q * u
}

/// The rho function from Imhof (1961), Eq 3.2.
fn rho(u: f64, params: &ImhofParams) -> f64 {
    let mut prod = 1.0;
    for &lam in params.lambda {
        prod *= (1.0 + lam.powi(2) * u.powi(2)).powf(0.25);
    }

    prod
}

/// The original integrand: sin(theta(u)) / (u * rho(u))
fn imhof_integrand(u: f64, params: &ImhofParams) -> f64 {
    if u.abs() < 1e-9 {
        // Limit as u -> 0
        let sum_lam: f64 = params.lambda.iter().sum();
        return 0.5 * (sum_lam - params.q);
    }

    let t = theta(u, params);
    let r = rho(u, params);

    t.sin() / (u * r)
}

/// Transformation: u = t / (1 - t)
fn transformed_integrand(t: f64, params: &ImhofParams) -> f64 {
    // Avoid singularity at t=1.0, though the limit is 0.
    if t >= 1.0 - 1e-9 {
        return 0.0;
    }

    let one_minus_t = 1.0 - t;
    let u = t / one_minus_t;
    let du_dt = 1.0 / one_minus_t.powi(2);

    imhof_integrand(u, params) * du_dt
}

/// Computes P(Q > q) using Imhof's method.
///
/// Formula: P(Q > x) = 1/2 + 1/pi * integral_0^inf (sin(theta(u)) / (u * rho(u))) du
pub(crate) fn significance_level(statistic: f64, eigenvalues: &[f64]) -> f64 {
    let params = ImhofParams {
        q: statistic,
        lambda: eigenvalues,
    };

    // We integrate the transformed function from 0 to 1.
    // This covers the original range of 0 to infinity.
    let result = NewtonCotes::new(|t| transformed_integrand(t, &params)).integrate(0.0, 1.0).unwrap();

    // P(Q > x)
    let p_val = 0.5 + result / PI;

    // Clamp to valid probability range [0, 1]
    p_val.clamp(0.0, 1.0)
}
