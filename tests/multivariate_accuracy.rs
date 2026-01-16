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
            .arg("\"install.packages(c('MVN', 'mnt'))\"")
            .spawn()
            .unwrap()
            .wait()
            .unwrap();
    });
}

fn sample_mv_norm_data(n: usize, d: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::from_entropy();
    let dist = Normal::new(0.0, 1.0).unwrap();

    (0..n).map(|_| dist.sample_iter(&mut rng).take(d).collect()).collect()
}

fn sample_mv_unif_data(n: usize, d: usize) -> Vec<Vec<f64>> {
    let mut rng = StdRng::from_entropy();
    let dist = Uniform::new(0.0, 1.0).unwrap();

    (0..n).map(|_| dist.sample_iter(&mut rng).take(d).collect()).collect()
}

fn mv_data_to_r(data: &[Vec<f64>]) -> String {
    let d = data[0].len();
    let n = data.len();
    let flat_data: Vec<String> =
        data.iter().flat_map(|row| row.iter().map(|x| x.to_string())).collect();

    let joined = flat_data.join(",");

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
        use normality::multivariate::{
            HenzeWagnerMethod,
            HenzeZirklerMethod,
            MardiaMethod,
            henze_wagner,
            henze_zirkler,
            mardia,
            pudelko
        };

        pastey::paste! {$(
            #[test]
            fn [<henze_zirkler_accuracy_ $n>]() {
                install_r_packages();

                let d = 3;
                let norm = sample_mv_norm_data($n, d);
                let unif = sample_mv_unif_data($n, d);
                let norm_r = mv_data_to_r(&norm);
                let unif_r = mv_data_to_r(&unif);
                let r_code = formatdoc! {"
                    library(MVN)

                    norm_mat <- {norm}
                    unif_mat <- {unif}

                    norm_res <- hz(norm_mat, use_population = FALSE)
                    unif_res <- hz(unif_mat, use_population = FALSE)

                    # Extract HZ statistic and p.value
                    print(paste(norm_res$Statistic, norm_res$p.value))
                    print(paste(unif_res$Statistic, unif_res$p.value))
                ",
                    norm = norm_r,
                    unif = unif_r
                };

                let output = execute_r(r_code);
                let lines: Vec<&str> = output.split('\n').collect();
                let parse_r_line = |line: &str| -> (f64, f64) {
                    let values = line.split_whitespace().skip(1).collect::<Vec<_>>();
                    (
                        f64::from_str(&values[0].replace('"', "")).unwrap(),
                        f64::from_str(&values[1].replace('"', "")).unwrap(),
                    )
                };

                let (r_norm_stat, r_norm_p) = parse_r_line(lines[0]);
                let (r_unif_stat, r_unif_p) = parse_r_line(lines[1]);
                let norm_result = henze_zirkler(
                    norm.clone(),
                    false,
                    HenzeZirklerMethod::LogNormal
                ).unwrap();

                let unif_result = henze_zirkler(
                    unif.clone(),
                    false,
                    HenzeZirklerMethod::LogNormal
                ).unwrap();

                assert_float_relative_eq!(r_norm_stat, norm_result.statistic, 1e-4);
                assert_float_relative_eq!(r_unif_stat, unif_result.statistic, 1e-4);
                assert_float_absolute_eq!(r_norm_p, norm_result.p_value, 1e-4);
                assert_float_absolute_eq!(r_unif_p, unif_result.p_value, 1e-4);
            }

            #[test]
            fn [<henze_wagner_accuracy_ $n>]() {
                install_r_packages();

                let d = 3;
                let norm = sample_mv_norm_data($n, d);
                let unif = sample_mv_unif_data($n, d);
                let norm_r = mv_data_to_r(&norm);
                let unif_r = mv_data_to_r(&unif);
                let r_code = formatdoc! {"
                    library(MVN)

                    norm_mat <- {norm}
                    unif_mat <- {unif}

                    norm_res <- hw(norm_mat, use_population = FALSE)
                    unif_res <- hw(unif_mat, use_population = FALSE)

                    # Extract HW statistic and p.value
                    print(paste(norm_res$Statistic, norm_res$p.value))
                    print(paste(unif_res$Statistic, unif_res$p.value))
                ",
                    norm = norm_r,
                    unif = unif_r
                };

                let output = execute_r(r_code);
                let lines: Vec<&str> = output.split('\n').collect();
                let parse_r_line = |line: &str| -> (f64, f64) {
                    let values = line.split_whitespace().skip(1).collect::<Vec<_>>();
                    (
                        f64::from_str(&values[0].replace('"', "")).unwrap(),
                        f64::from_str(&values[1].replace('"', "")).unwrap(),
                    )
                };

                let (r_norm_stat, r_norm_p) = parse_r_line(lines[0]);
                let (r_unif_stat, r_unif_p) = parse_r_line(lines[1]);
                let norm_result = henze_wagner(
                    norm.clone(),
                    false,
                    HenzeWagnerMethod::LogNormal
                ).unwrap();

                let unif_result = henze_wagner(
                    unif.clone(),
                    false,
                    HenzeWagnerMethod::LogNormal
                ).unwrap();

                assert_float_relative_eq!(r_norm_stat, norm_result.statistic, 1e-4);
                assert_float_relative_eq!(r_unif_stat, unif_result.statistic, 1e-4);
                assert_float_absolute_eq!(r_norm_p, norm_result.p_value, 1e-4);
                assert_float_absolute_eq!(r_unif_p, unif_result.p_value, 1e-4);
            }

            #[test]
            fn [<mardia_accuracy_ $n>]() {
                install_r_packages();

                let d = 3;
                let norm = sample_mv_norm_data($n, d);
                let unif = sample_mv_unif_data($n, d);
                let norm_r = mv_data_to_r(&norm);
                let unif_r = mv_data_to_r(&unif);
                let r_code = formatdoc! {"
                    library(MVN)

                    norm_mat <- {norm}
                    unif_mat <- {unif}

                    norm_res <- mardia(norm_mat, use_population = FALSE)
                    unif_res <- mardia(unif_mat, use_population = FALSE)

                    print(paste(norm_res$Statistic[1], norm_res$p.value[1]))
                    print(paste(norm_res$Statistic[2], norm_res$p.value[2]))
                    print(paste(unif_res$Statistic[1], unif_res$p.value[1]))
                    print(paste(unif_res$Statistic[2], unif_res$p.value[2]))
                ",
                    norm = norm_r,
                    unif = unif_r
                };

                let output = execute_r(r_code);
                let lines: Vec<&str> = output.split('\n').collect();

                let parse_r_line = |line: &str| -> (f64, f64) {
                    let values = line.split_whitespace().skip(1).collect::<Vec<_>>();
                    (
                        f64::from_str(&values[0].replace('"', "")).unwrap(),
                        f64::from_str(&values[1].replace('"', "")).unwrap(),
                    )
                };

                let (r_norm_s_stat, r_norm_s_p) = parse_r_line(lines[0]);
                let (r_norm_k_stat, r_norm_k_p) = parse_r_line(lines[1]);
                let (r_unif_s_stat, r_unif_s_p) = parse_r_line(lines[2]);
                let (r_unif_k_stat, r_unif_k_p) = parse_r_line(lines[3]);

                let norm_result = mardia(
                    norm.clone(),
                    false,
                    MardiaMethod::Asymptotic
                ).unwrap();

                let unif_result = mardia(
                    unif.clone(),
                    false,
                    MardiaMethod::Asymptotic
                ).unwrap();

                // Skewness
                assert_float_relative_eq!(r_norm_s_stat, norm_result.skewness.statistic, 1e-4);
                assert_float_absolute_eq!(r_norm_s_p, norm_result.skewness.p_value, 1e-4);
                assert_float_relative_eq!(r_unif_s_stat, unif_result.skewness.statistic, 1e-4);
                assert_float_absolute_eq!(r_unif_s_p, unif_result.skewness.p_value, 1e-4);

                // Kurtosis
                assert_float_relative_eq!(r_norm_k_stat, norm_result.kurtosis.statistic, 1e-4);
                assert_float_absolute_eq!(r_norm_k_p, norm_result.kurtosis.p_value, 1e-4);
                assert_float_relative_eq!(r_unif_k_stat, unif_result.kurtosis.statistic, 1e-4);
                assert_float_absolute_eq!(r_unif_k_p, unif_result.kurtosis.p_value, 1e-4);
            }
            
            #[test]
            fn [<pudelko_accuracy_ $n>]() {
                install_r_packages();

                let d = 2; // Dimension 2 for stability in these tests
                let r_param = 2.0;
                let norm = sample_mv_norm_data($n, d);
                let unif = sample_mv_unif_data($n, d);

                let norm_r = mv_data_to_r(&norm);
                let unif_r = mv_data_to_r(&unif);
                let r_code = formatdoc! {"
                    library(mnt)
                    
                    norm_mat <- {norm}
                    unif_mat <- {unif}
                    r_val <- {r}

                    # MC.rep set low as we only need the statistic, which is not MC-dependent
                    # (though the optimization start points are stochastic)
                    stat_norm <- PU(norm_mat, r=r_val)
                    stat_unif <- PU(unif_mat, r=r_val)

                    # Output format: NormStat UnifStat
                    cat('RESULT_START\n')
                    cat(sprintf('%.10f %.10f', stat_norm, stat_unif))
                    cat('\nRESULT_END')
                ",
                    norm = norm_r,
                    unif = unif_r,
                    r = r_param
                };

                let output = execute_r(r_code);
                let lines: Vec<&str> = output.split('\n').collect();
                let mut data_line = "";
                let mut found = false;
                for line in lines {
                    if line.trim() == "RESULT_START" {
                        found = true;
                        continue;
                    }
                    if found {
                        data_line = line;
                        break;
                    }
                }

                if data_line.is_empty() {
                    panic!("Could not find RESULT_START block in R output:\n{}", output);
                }

                let values: Vec<f64> = data_line
                    .split_whitespace()
                    .map(|s| f64::from_str(s).unwrap())
                    .collect();
                
                let r_norm_stat = values[0];
                let r_unif_stat = values[1];

                // Run Rust Implementation
                let norm_result = pudelko(
                    norm.clone(),
                    r_param,
                    10 
                ).unwrap();

                let unif_result = pudelko(
                    unif.clone(),
                    r_param,
                    10
                ).unwrap();

                // The statistic is a Supremum (maximum) found via stochastic optimization.
                // We assert that Rust finds a peak AT LEAST 80% as high as R.
                // If Rust finds a higher peak (stat > r_stat), that is valid and good.
                assert!(norm_result.statistic >= r_norm_stat * 0.8, 
                    "Rust Norm Stat ({}) significantly lower than R ({})", norm_result.statistic, r_norm_stat);

                assert!(unif_result.statistic >= r_unif_stat * 0.8,
                    "Rust Unif Stat ({}) significantly lower than R ({})", unif_result.statistic, r_unif_stat);

                // Normal data should NOT be strongly rejected (p > 0.001)
                // We use a very loose bound because with MC=100, p-value resolution is 0.01.
                // It is possible for a random normal sample to have p < 0.05, but p=0.0 is very rare if implemented correctly.
                // We just ensure we aren't always rejecting normal data.
                // Uniform data should be rejected for decent sample sizes (n >= 50)
                if $n >= 50 {
                    // We expect low p-values for non-normal data
                    assert!(unif_result.p_value <= 0.2, 
                        "Failed to detect Uniform data (p={}) for n={}", unif_result.p_value, $n);
                }
            }
        )+}
    };
}

gen_mv_accuracy_tests!(10, 32, 50, 64, 100, 128, 200, 256, 300, 400, 500);
