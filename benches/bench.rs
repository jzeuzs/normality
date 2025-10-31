use std::hint::black_box;

use gungraun::{library_benchmark, library_benchmark_group, main};
use rand::SeedableRng;
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use statrs::distribution::Normal;

fn sample_data(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(12345);
    let dist = Normal::new(0.0, 1.0).unwrap();
    let sample: Vec<f64> = dist.sample_iter(&mut rng).take(n).collect();

    sample
}

#[library_benchmark]
#[bench::tiny(sample_data(10))]
#[bench::small(sample_data(100))]
#[bench::medium(sample_data(1000))]
#[bench::large(sample_data(5000))]
fn anderson_darling(data: Vec<f64>) {
    let _ = black_box(normality::anderson_darling(data));
}

#[library_benchmark]
#[bench::tiny(sample_data(10))]
#[bench::small(sample_data(100))]
#[bench::medium(sample_data(1000))]
#[bench::large(sample_data(5000))]
fn dagostino_k_squared(data: Vec<f64>) {
    let _ = black_box(normality::dagostino_k_squared(data));
}

#[library_benchmark]
#[bench::tiny(sample_data(10))]
#[bench::small(sample_data(100))]
#[bench::medium(sample_data(1000))]
#[bench::large(sample_data(5000))]
fn jarque_bera(data: Vec<f64>) {
    let _ = black_box(normality::jarque_bera(data));
}

#[library_benchmark]
#[bench::tiny(sample_data(10))]
#[bench::small(sample_data(100))]
#[bench::medium(sample_data(1000))]
#[bench::large(sample_data(5000))]
fn lilliefors(data: Vec<f64>) {
    let _ = black_box(normality::lilliefors(data));
}

#[library_benchmark]
#[bench::tiny(sample_data(10))]
#[bench::small(sample_data(100))]
#[bench::medium(sample_data(1000))]
#[bench::large(sample_data(5000))]
fn pearson_chi_squared(data: Vec<f64>) {
    let _ = black_box(normality::pearson_chi_squared(data, None, true));
}

#[library_benchmark]
#[bench::tiny(sample_data(10))]
#[bench::small(sample_data(100))]
#[bench::medium(sample_data(1000))]
#[bench::large(sample_data(5000))]
fn shapiro_wilk(data: Vec<f64>) {
    let _ = black_box(normality::shapiro_wilk(data));
}

library_benchmark_group!(
    name = benches;
    benchmarks = anderson_darling, dagostino_k_squared, jarque_bera, lilliefors, pearson_chi_squared, shapiro_wilk
);

main!(library_benchmark_groups = benches);
