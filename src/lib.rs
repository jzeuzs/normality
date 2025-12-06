#![doc = include_str!("../README.md")]
#![warn(clippy::pedantic)]
#![allow(
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::many_single_char_names,
    clippy::similar_names,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    clippy::missing_panics_doc,
    clippy::missing_errors_doc
)]

mod error;
mod methods;

use std::iter::Sum;

pub use error::Error;
pub use methods::*;
use num_traits::{Float as Float_, Num, NumAssign, NumOps};

/// A convenience trait combining bounds frequently used for floating-point computations.
pub trait Float: Float_ + Num + NumAssign + NumOps + Sum {}

/// Blanket implementation of [`Float`] for any type that satisfies its bounds.
impl<T: Float_ + Num + NumAssign + NumOps + Sum> Float for T {}

/// A generic data structure to hold the results of a normality test.
///
/// This structure standardizes the output for various normality tests that
/// will be part of this crate.
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Computation<T: Float> {
    /// The calculated test statistic.
    pub statistic: T,

    /// The p-value corresponding to the test statistic. It indicates the probability
    /// of observing the given result, or one more extreme, if the null hypothesis is true.
    pub p_value: T,
}
