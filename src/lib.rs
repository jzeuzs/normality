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

#[macro_use]
pub(crate) mod macros;

mod error;
mod methods;

use std::iter::Sum;

pub use error::Error;
pub use methods::*;
use num_traits::{Float as Float_, Num, NumAssign, NumOps};

/// A convenience trait combining bounds frequently used for floating-point computations.
#[cfg(feature = "parallel")]
pub trait Float: Float_ + Num + NumAssign + NumOps + Sum + Send + Sync {}

/// Blanket implementation of [`Float`] for any type that satisfies its bounds.
#[cfg(feature = "parallel")]
impl<T: Float_ + Num + NumAssign + NumOps + Sum + Send + Sync> Float for T {}

/// A convenience trait combining bounds frequently used for floating-point computations.
#[cfg(not(feature = "parallel"))]
pub trait Float: Float_ + Num + NumAssign + NumOps + Sum {}

/// Blanket implementation of [`Float`] for any type that satisfies its bounds.
#[cfg(not(feature = "parallel"))]
impl<T: Float_ + Num + NumAssign + NumOps + Sum> Float for T {}

/// A generic data structure to hold the results of a normality test.
///
/// This structure standardizes the output for various normality tests that
/// will be part of this crate.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Computation<T: Float> {
    /// The calculated test statistic.
    pub statistic: T,

    /// The p-value corresponding to the test statistic. It indicates the probability
    /// of observing the given result, or one more extreme, if the null hypothesis is true.
    pub p_value: T,
}

#[cfg(all(feature = "serde", test))]
mod computation_serde_test {
    use serde_test::{Token, assert_ser_tokens};

    use super::Computation;

    #[test]
    fn test_computation_tokens() {
        let computation = Computation {
            statistic: 1.0,
            p_value: 0.05,
        };

        let expected_tokens = vec![
            Token::Struct {
                name: "Computation",
                len: 2,
            },
            Token::Str("statistic"),
            Token::F64(1.0),
            Token::Str("p_value"),
            Token::F64(0.05),
            Token::StructEnd,
        ];

        assert_ser_tokens(&computation, &expected_tokens);
    }
}
