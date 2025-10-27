use divan::{Bencher, bench, black_box};
use rand::prelude::Distribution;
use statrs::distribution::Normal;

fn sample_data(n: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let dist = Normal::new(0.0, 1.0).unwrap();
    let sample: Vec<f64> = dist.sample_iter(&mut rng).take(n).collect();

    sample
}

fn main() {
    divan::main();
}

#[bench]
fn shapiro_wilk(bencher: Bencher) {
    bencher
        .with_inputs(|| sample_data(10))
        .bench_values(|data| black_box(normality::shapiro_wilk(data)));
}

#[bench]
fn lilliefors(bencher: Bencher) {
    bencher
        .with_inputs(|| sample_data(10))
        .bench_values(|data| black_box(normality::lilliefors(data)));
}
