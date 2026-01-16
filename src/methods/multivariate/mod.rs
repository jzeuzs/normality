//! Multivariate normality tests.
//!
//! This module provides several statistical tests to assess whether a dataset follows a
//! multivariate normal distribution.
//!
//! # Example
//!
//! ```rust
//! use normality::multivariate::{HenzeZirklerMethod, henze_zirkler};
//!
//! let data = vec![vec![0.1, 0.2], vec![0.5, 0.1], vec![-0.2, 0.3], vec![0.8, -0.5]];
//!
//! // Perform the Henze-Zirkler test
//! let result = henze_zirkler(data, true, HenzeZirklerMethod::LogNormal).unwrap();
//! assert!(result.p_value > 0.05);
//! ```

mod henze_wagner;
mod henze_zirkler;
mod mardia;
mod pudelko;

pub use henze_wagner::{HenzeWagnerMethod, henze_wagner};
pub use henze_zirkler::{HenzeZirklerMethod, henze_zirkler};
pub use mardia::{MardiaComputation, MardiaMethod, mardia};
pub use pudelko::pudelko;
