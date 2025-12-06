use thiserror::Error as ThisError;

/// Represents errors that can occur during a normality test computation.
#[derive(Debug, ThisError, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The input sample size is too small for the test.
    #[error("Sample size must be at least {needed}, but was given {given}.")]
    InsufficientSampleSize { given: usize, needed: usize },

    /// The input sample size is too big for the test.
    #[error("Sample size must be at most {needed}, but was given {given}.")]
    ExcessiveSampleSize { given: usize, needed: usize },

    /// The range of the input data is zero (i.e., all values are the same).
    /// This prevents the calculation of the test statistic.
    #[error("The range of the data is zero, the test cannot be computed.")]
    ZeroRange,

    /// The input data contains `NaN` values.
    /// Normality tests cannot be performed on data with `NaN`s.
    #[error("Input data must not contain NaN values.")]
    ContainsNaN,

    /// See [`statrs::distribution::NormalError`].
    #[error("{0}")]
    NormalDistributionError(#[from] statrs::distribution::NormalError),

    /// See [`statrs::distribution::GammaError`].
    #[error("{0}")]
    GammaError(#[from] statrs::distribution::GammaError),

    /// Other errors.
    #[error("{0}")]
    Other(String),
}
