use tdigests::naive_quantile;
use tdigests::naive_rank;
use tdigests::TDigest;

fn main() {
    // Simulate data points (e.g., latencies, measurements)
    let values = (0..=10000).map(|i| i as f64 / 100.0).collect::<Vec<f64>>();

    // Create a new t-digest
    let mut digest = TDigest::from_values(values.clone());

    // Compress the t-digest with a maximum number of centroids
    digest.compress(10);

    // Estimate quantiles
    let p50 = digest.estimate_quantile(0.5);
    let p75 = digest.estimate_quantile(0.75);
    let p90 = digest.estimate_quantile(0.9);
    let p99 = digest.estimate_quantile(0.99);
    let rank = digest.estimate_rank(50.0);

    println!("estimated 50th percentile (median): {:.3}", p50);
    println!("estimated 75th percentile: {:.3}", p75);
    println!("estimated 90th percentile: {:.3}", p90);
    println!("estimated 99th percentile: {:.3}", p99);
    println!("estimated rank of 50.0: {:.3}", rank);

    // Compare with naive quantile estimation
    let naive_p50 = naive_quantile(&values, 0.5);
    let naive_p75 = naive_quantile(&values, 0.75);
    let naive_p90 = naive_quantile(&values, 0.9);
    let naive_p99 = naive_quantile(&values, 0.99);
    let naive_rank = naive_rank(&values, 50.0);

    println!("naive 50th percentile (median): {:.3}", naive_p50);
    println!("naive 75th percentile: {:.3}", naive_p75);
    println!("naive 90th percentile: {:.3}", naive_p90);
    println!("naive 99th percentile: {:.3}", naive_p99);
    println!("naive rank of 50.0: {:.3}", naive_rank);

    println!(
        "number of centroids after compression:: {}",
        digest.centroids().len()
    );
}
