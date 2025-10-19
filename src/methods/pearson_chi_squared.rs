use statrs::distribution::{ChiSquared, ContinuousCDF, Normal};

use crate::{Computation, Error, Float};

/// Performs the Pearson chi-squared test for normality.
///
/// The test assesses normality by binning the data and comparing the observed
/// frequency in each bin to the expected frequency under a normal distribution.
///
/// Takes an argument `data` which is a slice (`&[T]`) containing the data sample. This can be a
/// [`Vec<T>`](Vec), an array [`[T; N]`](std::slice), or an [`ndarray::Array1<T>`](ndarray::Array1).
///
/// It also takes an `n_classes` which is an optional [`usize`] specifying the number of classes
/// (bins) to use. If [`None`], the number of classes is determined by the formula: `ceil(2 *
/// n^(2/5))`.
///
/// Finally, it takes another argument `adjust` which indicates whether to adjust the degrees of
/// freedom for the fact that the mean and standard deviation are estimated from the data. If
/// `true`, 2 is subtracted from the degrees of freedom.
///
/// The sample size of `data` must be more than 1.
/// Also, the range of `data` must not be equal to 0.
pub fn pearson_chi_squared<T: Float>(
    data: &[T],
    n_classes: Option<usize>,
    adjust: bool,
) -> Result<Computation<T>, Error> {
    let clean_data: Vec<T> = data.iter().copied().filter(|v| !v.is_nan()).collect();
    let n = clean_data.len();

    if n < 2 {
        return Err(Error::InsufficientSampleSize {
            given: n,
            needed: 2,
        });
    }

    let n_t = T::from(n).unwrap();

    // Determine the number of classes for the histogram.
    let num_classes = n_classes.unwrap_or_else(|| (2.0 * (n as f64).powf(0.4)).ceil() as usize);
    let num_classes_t = T::from(num_classes).unwrap();

    // Calculate sample mean and standard deviation.
    let mean = clean_data.iter().copied().sum::<T>() / n_t;
    let variance =
        clean_data.iter().map(|&x| (x - mean).powi(2)).sum::<T>() / T::from(n - 1).unwrap();

    let std_dev = variance.sqrt();

    if std_dev < T::epsilon() {
        return Err(Error::ZeroRange);
    }

    // Bin the data based on the normal distribution's CDF.
    let normal_dist = Normal::new(mean.to_f64().unwrap(), std_dev.to_f64().unwrap())?;
    let mut counts = vec![0; num_classes];

    for &x in &clean_data {
        let p = normal_dist.cdf(x.to_f64().unwrap());
        let bin_num = (1.0 + num_classes_t.to_f64().unwrap() * p).floor() as usize;
        if bin_num >= 1 && bin_num <= num_classes {
            counts[bin_num - 1] += 1; // Convert 1-based bin to 0-based index.
        }
    }

    // Calculate the chi-squared statistic.
    let expected_count = n_t / num_classes_t;
    let chi_sq_stat = counts
        .iter()
        .map(|&count| {
            let count_t = T::from(count).unwrap();
            let diff = count_t - expected_count;
            diff.powi(2) / expected_count
        })
        .sum::<T>();

    // Calculate degrees of freedom.
    let dfd = if adjust { 2 } else { 0 };
    let df = num_classes.saturating_sub(dfd).saturating_sub(1);

    if df < 1 {
        // The test is not valid if degrees of freedom is less than 1.
        return Err(Error::Other("Degrees of freedom is less than 1".to_string())); // Using a generic code for invalid df.
    }

    // Calculate p-value from the chi-squared distribution.
    let chi_sq_dist = ChiSquared::new(df as f64)?;
    let p_value = T::from(chi_sq_dist.sf(chi_sq_stat.to_f64().unwrap())).unwrap();

    Ok(Computation {
        statistic: chi_sq_stat,
        p_value,
    })
}
