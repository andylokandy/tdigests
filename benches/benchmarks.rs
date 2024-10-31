use divan::Bencher;
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tdigests::TDigest;

fn generate_digest(n: usize) -> TDigest {
    let mut rng = StdRng::seed_from_u64(42);
    let uniform = Uniform::new(0.0, 1.0);
    let values = (0..n).map(|_| uniform.sample(&mut rng)).collect::<Vec<_>>();
    TDigest::from_values(values)
}

#[divan::bench(args = [100, 1000, 10000, 100000, 1000000])]
fn benchmark_from_values(bencher: Bencher, n: usize) {
    bencher
        .with_inputs(|| (0..n).map(|v| v as f64).collect::<Vec<_>>())
        .bench_values(TDigest::from_values);
}

#[divan::bench(args = [100, 1000, 10000, 100000, 1000000])]
fn benchmark_compress(bencher: Bencher, n: usize) {
    bencher
        .with_inputs(|| generate_digest(n))
        .bench_values(|mut digest| {
            digest.compress(100);
            digest
        });
}

#[divan::bench(args = [100, 1000, 10000, 100000, 1000000])]
fn benchmark_estimate_quantile(bencher: Bencher, n: usize) {
    bencher
        .with_inputs(|| {
            let mut digest = generate_digest(n * 2);
            digest.compress(n);
            digest
        })
        .bench_values(|digest| digest.estimate_quantile(0.5));
}

#[divan::bench(args = [100, 1000, 10000, 100000, 1000000])]
fn benchmark_estimate_rank(bencher: Bencher, n: usize) {
    bencher
        .with_inputs(|| {
            let mut digest = generate_digest(n * 2);
            digest.compress(n);
            digest
        })
        .bench_values(|digest| digest.estimate_rank(0.5));
}

fn main() {
    divan::main();
}
