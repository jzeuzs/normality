use std::env;
use std::fs::remove_dir_all;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::str::FromStr;
use std::sync::Once;

use assert_float_eq::{assert_float_absolute_eq, assert_float_relative_eq};
use indoc::formatdoc;
use nanoid::nanoid;
use normality::multivariate::{HenzeZirklerMethod, henze_zirkler};
use rand::SeedableRng;
use rand::distributions::Distribution;
use rand::rngs::StdRng;
use statrs::distribution::{Normal, Uniform};
use tempfile::{Builder, tempdir};

static INIT: Once = Once::new();

fn install_r_packages() {
    INIT.call_once(|| {
        Command::new("Rscript")
            .arg("-e")
            // Install MVN for multivariate tests
            .arg("\"install.packages(c('MVN'))\"")
            .spawn()
            .unwrap()
            .wait()
            .unwrap();
    });
}

/// Generates N observations of D-dimensional multivariate normal data (uncorrelated).
fn sample_mv_norm_data(n: usize, d: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::from_entropy();
    let dist = Normal::new(0.0, 1.0).unwrap();

    (0..n).map(|_| dist.sample_iter(&mut rng).take(d).collect()).collect()
}

/// Generates N observations of D-dimensional independent uniform data.
fn sample_mv_unif_data(n: usize, d: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::from_entropy();
    let dist = Uniform::new(0.0, 1.0).unwrap();

    (0..n).map(|_| dist.sample_iter(&mut rng).take(d).collect()).collect()
}

/// Converts a Vec<Vec<f64>> into an R matrix string string.
fn mv_data_to_r(data: &[Vec<f64>]) -> String {
    let d = data[0].len();
    let n = data.len();

    // Flatten data for R's c() function
    let flat_data: Vec<String> =
        data.iter().flat_map(|row| row.iter().map(|x| x.to_string())).collect();

    let joined = flat_data.join(",");

    // Construct R matrix: matrix(c(...), nrow=n, ncol=d, byrow=TRUE)
    format!("matrix(c({joined}), nrow={n}, ncol={d}, byrow=TRUE)")
}

fn execute_r(code: String) -> String {
    let temp_dir = env::var("TEMP_DIR").map(PathBuf::from).unwrap_or_else(|_| {
        let temp_dir = tempdir().unwrap();
        temp_dir.keep()
    });

    let mut temp_file = Builder::new()
        .prefix(&format!("normalityrs-mv-test-{}", nanoid!()))
        .suffix(".R")
        .tempfile_in(&temp_dir)
        .unwrap();

    writeln!(temp_file, "{}", code).unwrap();

    let path = temp_file.path();
    let output = Command::new("Rscript").arg(path).output().unwrap();

    if temp_dir.to_string_lossy() != env::var("TEMP_DIR").unwrap_or_else(|_| String::new()) {
        remove_dir_all(temp_dir).unwrap();
    }

    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

macro_rules! gen_mv_accuracy_tests {
    ($($n:expr),+) => {
        pastey::paste! {$(
            #[test]
            fn [<henze_zirkler_accuracy_ $n>]() {
                install_r_packages();

                // Test with d=3 dimensions
                let d = 3;
                let norm = sample_mv_norm_data($n, d);
                let unif = sample_mv_unif_data($n, d);

                let norm_r = mv_data_to_r(&norm);
                let unif_r = mv_data_to_r(&unif);

                // MVN::hz returns a dataframe with columns: Test, HZ, p.value
                let r_code = formatdoc! {"
                    library(MVN)

                    norm_mat <- {norm}
                    unif_mat <- {unif}

                    norm_res <- hz(norm_mat, use_population = TRUE)
                    unif_res <- hz(unif_mat, use_population = TRUE)

                    # Extract HZ statistic and p.value
                    print(paste(norm_res$Statistic, norm_res$p.value))
                    print(paste(unif_res$Statistic, unif_res$p.value))
                ",
                    norm = norm_r,
                    unif = unif_r
                };

                let output = execute_r(r_code);
                let lines: Vec<&str> = output.split('\n').collect();

                // Parse R Output
                // Expected format: "[1] \"<stat> <pvalue>\""
                let parse_r_line = |line: &str| -> (f64, f64) {
                    let values = line.split_whitespace().skip(1).collect::<Vec<_>>();
                    (
                        f64::from_str(&values[0].replace('"', "")).unwrap(),
                        f64::from_str(&values[1].replace('"', "")).unwrap(),
                    )
                };

                let (r_norm_stat, r_norm_p) = parse_r_line(lines[0]);
                let (r_unif_stat, r_unif_p) = parse_r_line(lines[1]);

                // Run Rust Implementation
                // use_population_covariance = true, Method = LogNormal
                let norm_result = henze_zirkler(
                    norm.clone(),
                    true,
                    HenzeZirklerMethod::LogNormal
                ).unwrap();

                let unif_result = henze_zirkler(
                    unif.clone(),
                    true,
                    HenzeZirklerMethod::LogNormal
                ).unwrap();

                // Assertions
                // HZ Statistic should be very close
                assert_float_relative_eq!(r_norm_stat, norm_result.statistic, 1e-4);
                assert_float_relative_eq!(r_unif_stat, unif_result.statistic, 1e-4);

                // P-value matches
                // Note: Small floating point diffs in the log-normal approx can occur,
                // so we use a slightly looser relative epsilon or absolute eq for very small ps.
                assert_float_absolute_eq!(r_norm_p, norm_result.p_value, 1e-4);
                assert_float_absolute_eq!(r_unif_p, unif_result.p_value, 1e-4);
            }
        )+}
    };
}

// Run tests for various sample sizes
gen_mv_accuracy_tests!(10, 32, 50, 64, 100, 128, 200, 256, 300, 400, 500);
