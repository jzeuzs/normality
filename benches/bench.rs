use std::hint::black_box;
use std::sync::LazyLock;

use gungraun::{library_benchmark, library_benchmark_group, main};
use rand::SeedableRng;
use rand::prelude::Distribution;
use rand::rngs::StdRng;
use statrs::distribution::Normal;

const SEED: u64 = 123;
static TINY: LazyLock<Vec<f64>> = LazyLock::new(|| sample_data(10));
static SMALL: LazyLock<Vec<f64>> = LazyLock::new(|| sample_data(100));
static MEDIUM: LazyLock<Vec<f64>> = LazyLock::new(|| sample_data(1000));
static LARGE: LazyLock<Vec<f64>> = LazyLock::new(|| sample_data(5000));

fn sample_data(n: usize) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(SEED);
    let dist = Normal::new(0.0, 1.0).unwrap();
    let sample: Vec<f64> = dist.sample_iter(&mut rng).take(n).collect();

    sample
}

fn to_vec(data: &LazyLock<Vec<f64>>) -> Vec<f64> {
    (*data).clone()
}

fn setup() {
    let _ = TINY;
    let _ = SMALL;
    let _ = MEDIUM;
    let _ = LARGE;
}

#[library_benchmark(setup = to_vec)]
#[bench::tiny(&TINY)]
#[bench::small(&SMALL)]
#[bench::medium(&MEDIUM)]
#[bench::large(&LARGE)]
fn anderson_darling(data: Vec<f64>) {
    let _ = black_box(normality::anderson_darling(data));
}

#[library_benchmark(setup = to_vec)]
#[bench::tiny(&TINY)]
#[bench::small(&SMALL)]
#[bench::medium(&MEDIUM)]
#[bench::large(&LARGE)]
fn anscombe_glynn(data: Vec<f64>) {
    let _ = black_box(normality::anscombe_glynn(data));
}

#[library_benchmark(setup = to_vec)]
#[bench::tiny(&TINY)]
#[bench::small(&SMALL)]
#[bench::medium(&MEDIUM)]
#[bench::large(&LARGE)]
fn dagostino_k_squared(data: Vec<f64>) {
    let _ = black_box(normality::dagostino_k_squared(data));
}

#[library_benchmark(setup = to_vec)]
#[bench::tiny(&TINY)]
#[bench::small(&SMALL)]
#[bench::medium(&MEDIUM)]
#[bench::large(&LARGE)]
fn jarque_bera(data: Vec<f64>) {
    let _ = black_box(normality::jarque_bera(data));
}

#[library_benchmark(setup = to_vec)]
#[bench::tiny(&TINY)]
#[bench::small(&SMALL)]
#[bench::medium(&MEDIUM)]
#[bench::large(&LARGE)]
fn lilliefors(data: Vec<f64>) {
    let _ = black_box(normality::lilliefors(data));
}

#[library_benchmark(setup = to_vec)]
#[bench::tiny(&TINY)]
#[bench::small(&SMALL)]
#[bench::medium(&MEDIUM)]
#[bench::large(&LARGE)]
fn pearson_chi_squared(data: Vec<f64>) {
    let _ = black_box(normality::pearson_chi_squared(data, None, true));
}

#[library_benchmark(setup = to_vec)]
#[bench::tiny(&TINY)]
#[bench::small(&SMALL)]
#[bench::medium(&MEDIUM)]
#[bench::large(&LARGE)]
fn shapiro_wilk(data: Vec<f64>) {
    let _ = black_box(normality::shapiro_wilk(data));
}

library_benchmark_group!(
    name = benches;
    setup = setup();
    benchmarks = anderson_darling, anscombe_glynn, dagostino_k_squared, jarque_bera, lilliefors, pearson_chi_squared, shapiro_wilk
);

main!(library_benchmark_groups = benches);
