use std::io::Write;
use std::process::Command;

use nanoid::nanoid;
use rand::distributions::Distribution;
use statrs::distribution::{Normal, Uniform};
use tempfile::Builder;

fn sample_norm_data(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let dist = Normal::new(0.0, 1.0).unwrap();
    let sample: Vec<f64> = dist.sample_iter(&mut rng).take(n).collect();

    sample
}

fn sample_unif_data(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let dist = Uniform::new(0.0, 1.0).unwrap();
    let sample: Vec<f64> = dist.sample_iter(&mut rng).take(n).collect();

    sample
}

fn data_to_r(data: &[f64]) -> String {
    let joined = data.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",");

    format!("c({joined})")
}

fn execute_r(code: String) -> String {
    let mut temp_file = Builder::new()
        .prefix(&format!("normalityrs-test-{}", nanoid!()))
        .suffix(".R")
        .tempfile_in("./")
        .unwrap();

    writeln!(temp_file, "{}", code).unwrap();

    let path = temp_file.path();
    let output = Command::new("Rscript").arg(path).output().unwrap();

    String::from_utf8_lossy(&output.stdout).trim().to_string()
}

macro_rules! gen_accuracy_tests {
    ($($n:expr),+) => {
        use std::str::FromStr;

        use assert_float_eq::{assert_float_absolute_eq, assert_float_relative_eq};
        use indoc::formatdoc;
        use normality::{
            anderson_darling,
            dagostino_k_squared,
            jarque_bera,
            lilliefors,
            pearson_chi_squared,
            shapiro_wilk,
        };

        pastey::paste! {$(
            #[test]
            fn [<shapiro_wilk_accuracy_ $n>]() {
                let norm = sample_norm_data($n);
                let unif = sample_unif_data($n);

                let norm_r = data_to_r(&norm);
                let unif_r = data_to_r(&unif);

                let r_code = formatdoc! {"
                    norm <- {norm}
                    unif <- {unif}

                    norm_result <- shapiro.test(norm)
                    unif_result <- shapiro.test(unif)

                    print(paste(norm_result$statistic, norm_result$p.value))
                    print(paste(unif_result$statistic, unif_result$p.value))
                ",
                    norm = norm_r,
                    unif = unif_r
                };

                let [(r_norm_stat, r_norm_p), (r_unif_stat, r_unif_p), ..] = execute_r(r_code)
                    .split("\n")
                    .map(|line| {
                        let values = line.split_whitespace().skip(1).collect::<Vec<_>>();

                        (
                            f64::from_str(&values[0].replace('"', "")).unwrap(),
                            f64::from_str(&values[1].replace('"', "")).unwrap(),
                        )
                    })
                    .collect::<Vec<_>>()[..]
                else {
                    unreachable!()
                };

                let norm_result = shapiro_wilk(&norm).unwrap();
                let unif_result = shapiro_wilk(&unif).unwrap();

                assert_float_absolute_eq!(r_norm_stat, norm_result.statistic);
                assert_float_absolute_eq!(r_norm_p, norm_result.p_value);
                assert_float_absolute_eq!(r_unif_stat, unif_result.statistic);
                assert_float_absolute_eq!(r_unif_p, unif_result.p_value);

                assert_float_relative_eq!(r_norm_stat, norm_result.statistic, 1e-4);
                assert_float_relative_eq!(r_norm_p, norm_result.p_value, 1e-4);
                assert_float_relative_eq!(r_unif_stat, unif_result.statistic, 1e-4);
                assert_float_relative_eq!(r_unif_p, unif_result.p_value, 1e-4);
            }

            #[test]
            fn [<lilliefors_accuracy_ $n>]() {
                let norm = sample_norm_data($n);
                let unif = sample_unif_data($n);

                let norm_r = data_to_r(&norm);
                let unif_r = data_to_r(&unif);

                let r_code = formatdoc! {"
                    library(nortest)

                    norm <- {norm}
                    unif <- {unif}

                    norm_result <- lillie.test(norm)
                    unif_result <- lillie.test(unif)

                    print(paste(norm_result$statistic, norm_result$p.value))
                    print(paste(unif_result$statistic, unif_result$p.value))
                ",
                    norm = norm_r,
                    unif = unif_r
                };

                let [(r_norm_stat, r_norm_p), (r_unif_stat, r_unif_p), ..] = execute_r(r_code)
                    .split("\n")
                    .map(|line| {
                        let values = line.split_whitespace().skip(1).collect::<Vec<_>>();

                        (
                            f64::from_str(&values[0].replace('"', "")).unwrap(),
                            f64::from_str(&values[1].replace('"', "")).unwrap(),
                        )
                    })
                    .collect::<Vec<_>>()[..]
                else {
                    unreachable!()
                };

                let norm_result = lilliefors(&norm).unwrap();
                let unif_result = lilliefors(&unif).unwrap();

                assert_float_absolute_eq!(r_norm_stat, norm_result.statistic);
                assert_float_absolute_eq!(r_norm_p, norm_result.p_value);
                assert_float_absolute_eq!(r_unif_stat, unif_result.statistic);
                assert_float_absolute_eq!(r_unif_p, unif_result.p_value);

                assert_float_relative_eq!(r_norm_stat, norm_result.statistic, 1e-4);
                assert_float_relative_eq!(r_norm_p, norm_result.p_value, 1e-4);
                assert_float_relative_eq!(r_unif_stat, unif_result.statistic, 1e-4);
                assert_float_relative_eq!(r_unif_p, unif_result.p_value, 1e-4);
            }

            #[test]
            fn [<anderson_darling_accuracy_ $n>]() {
                let norm = sample_norm_data($n);
                let unif = sample_unif_data($n);

                let norm_r = data_to_r(&norm);
                let unif_r = data_to_r(&unif);

                let r_code = formatdoc! {"
                    library(nortest)

                    norm <- {norm}
                    unif <- {unif}

                    norm_result <- ad.test(norm)
                    unif_result <- ad.test(unif)

                    print(paste(norm_result$statistic, norm_result$p.value))
                    print(paste(unif_result$statistic, unif_result$p.value))
                ",
                    norm = norm_r,
                    unif = unif_r
                };

                let [(r_norm_stat, r_norm_p), (r_unif_stat, r_unif_p), ..] = execute_r(r_code)
                    .split("\n")
                    .map(|line| {
                        let values = line.split_whitespace().skip(1).collect::<Vec<_>>();

                        (
                            f64::from_str(&values[0].replace('"', "")).unwrap(),
                            f64::from_str(&values[1].replace('"', "")).unwrap(),
                        )
                    })
                    .collect::<Vec<_>>()[..]
                else {
                    unreachable!()
                };

                let norm_result = anderson_darling(&norm).unwrap();
                let unif_result = anderson_darling(&unif).unwrap();

                assert_float_absolute_eq!(r_norm_stat, norm_result.statistic);
                assert_float_absolute_eq!(r_norm_p, norm_result.p_value);
                assert_float_absolute_eq!(r_unif_stat, unif_result.statistic);
                assert_float_absolute_eq!(r_unif_p, unif_result.p_value);

                assert_float_relative_eq!(r_norm_stat, norm_result.statistic, 1e-4);
                assert_float_relative_eq!(r_norm_p, norm_result.p_value, 1e-4);
                assert_float_relative_eq!(r_unif_stat, unif_result.statistic, 1e-4);
                assert_float_relative_eq!(r_unif_p, unif_result.p_value, 1e-4);
            }

            #[test]
            fn [<jarque_bera_accuracy_ $n>]() {
                let norm = sample_norm_data($n);
                let unif = sample_unif_data($n);

                let norm_r = data_to_r(&norm);
                let unif_r = data_to_r(&unif);

                let r_code = formatdoc! {"
                    library(moments)

                    norm <- {norm}
                    unif <- {unif}

                    norm_result <- jarque.test(norm)
                    unif_result <- jarque.test(unif)

                    print(paste(norm_result$statistic, norm_result$p.value))
                    print(paste(unif_result$statistic, unif_result$p.value))
                ",
                    norm = norm_r,
                    unif = unif_r
                };

                let [(r_norm_stat, r_norm_p), (r_unif_stat, r_unif_p), ..] = execute_r(r_code)
                    .split("\n")
                    .map(|line| {
                        let values = line.split_whitespace().skip(1).collect::<Vec<_>>();

                        (
                            f64::from_str(&values[0].replace('"', "")).unwrap(),
                            f64::from_str(&values[1].replace('"', "")).unwrap(),
                        )
                    })
                    .collect::<Vec<_>>()[..]
                else {
                    unreachable!()
                };

                let norm_result = jarque_bera(&norm).unwrap();
                let unif_result = jarque_bera(&unif).unwrap();

                assert_float_absolute_eq!(r_norm_stat, norm_result.statistic);
                assert_float_absolute_eq!(r_norm_p, norm_result.p_value);
                assert_float_absolute_eq!(r_unif_stat, unif_result.statistic);
                assert_float_absolute_eq!(r_unif_p, unif_result.p_value);
            }

            #[test]
            fn [<dagostino_k_squared_accuracy_ $n>]() {
                let norm = sample_norm_data($n);
                let unif = sample_unif_data($n);

                let norm_r = data_to_r(&norm);
                let unif_r = data_to_r(&unif);

                let r_code = formatdoc! {"
                    library(moments)

                    norm <- {norm}
                    unif <- {unif}

                    norm_result <- agostino.test(norm)
                    unif_result <- agostino.test(unif)

                    print(paste(norm_result$statistic[[2]], norm_result$p.value))
                    print(paste(unif_result$statistic[[2]], unif_result$p.value))
                ",
                    norm = norm_r,
                    unif = unif_r
                };

                let [(r_norm_stat, r_norm_p), (r_unif_stat, r_unif_p), ..] = execute_r(r_code)
                    .split("\n")
                    .map(|line| {
                        let values = line.split_whitespace().skip(1).collect::<Vec<_>>();

                        (
                            f64::from_str(&values[0].replace('"', "")).unwrap(),
                            f64::from_str(&values[1].replace('"', "")).unwrap(),
                        )
                    })
                    .collect::<Vec<_>>()[..]
                else {
                    unreachable!()
                };

                let norm_result = dagostino_k_squared(&norm).unwrap();
                let unif_result = dagostino_k_squared(&unif).unwrap();

                assert_float_absolute_eq!(r_norm_stat, norm_result.statistic);
                assert_float_absolute_eq!(r_norm_p, norm_result.p_value);
                assert_float_absolute_eq!(r_unif_stat, unif_result.statistic);
                assert_float_absolute_eq!(r_unif_p, unif_result.p_value);
            }

            #[test]
            fn [<pearson_chi_squared_accuracy_ $n>]() {
                let norm = sample_norm_data($n);
                let unif = sample_unif_data($n);

                let norm_r = data_to_r(&norm);
                let unif_r = data_to_r(&unif);

                let r_code = formatdoc! {"
                    library(nortest)

                    norm <- {norm}
                    unif <- {unif}

                    norm_result <- pearson.test(norm)
                    unif_result <- pearson.test(unif)

                    print(paste(norm_result$statistic[[1]], norm_result$p.value))
                    print(paste(unif_result$statistic[[1]], unif_result$p.value))
                ",
                    norm = norm_r,
                    unif = unif_r
                };

                let [(r_norm_stat, r_norm_p), (r_unif_stat, r_unif_p), ..] = execute_r(r_code)
                    .split("\n")
                    .map(|line| {
                        let values = line.split_whitespace().skip(1).collect::<Vec<_>>();

                        (
                            f64::from_str(&values[0].replace('"', "")).unwrap(),
                            f64::from_str(&values[1].replace('"', "")).unwrap(),
                        )
                    })
                    .collect::<Vec<_>>()[..]
                else {
                    unreachable!()
                };

                let norm_result = pearson_chi_squared(&norm, None, true).unwrap();
                let unif_result = pearson_chi_squared(&unif, None, true).unwrap();

                assert_float_absolute_eq!(r_norm_stat, norm_result.statistic);
                assert_float_absolute_eq!(r_norm_p, norm_result.p_value);
                assert_float_absolute_eq!(r_unif_stat, unif_result.statistic);
                assert_float_absolute_eq!(r_unif_p, unif_result.p_value);
            }
        )+}
    };
}

gen_accuracy_tests!(
    10, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 3500, 4000,
    4500, 5000
);
