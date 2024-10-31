use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::rngs::StdRng;
use rand::SeedableRng;
use tdigests::naive_quantile;
use tdigests::TDigest;

fn main() {
    // Simulate data points (e.g., latencies, measurements)
    let mut rng = StdRng::seed_from_u64(42);
    let uniform = Uniform::new(0.0, 100.0);
    let mut values: Vec<f64> = (0..100000).map(|_| uniform.sample(&mut rng)).collect();

    // Create a new t-digest
    let mut digest = TDigest::from_values(values.clone());

    // Compress the t-digest with a maximum number of centroids
    digest.compress(10);

    // Sort the values to speed up naive quantile estimation
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mut sample_count = 0;
    let mut cumulative_error = 0.0;
    for p in (0..=1000).map(|i| i as f64 / 1000.0) {
        let p_digest = digest.estimate_quantile(p);
        let p_naive = naive_quantile(&values, p);
        let error = (p_digest - p_naive).abs();
        cumulative_error += error;
        sample_count += 1;

        println!(
            "p: {:.3}, estimated: {:.3}, expected: {:.3}, error: {:.3}",
            p, p_digest, p_naive, error
        );
    }

    println!(
        "average error: {:.3}%",
        cumulative_error / sample_count as f64
    );
}
