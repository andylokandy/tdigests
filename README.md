# T-Digest Rust Library

[![Crates.io](https://img.shields.io/crates/v/tdigests)](https://crates.io/crates/tdigests)
[![Docs.rs](https://docs.rs/tdigests/badge.svg)](https://docs.rs/tdigests)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An efficient Rust implementation of the [t-digest algorithm](https://github.com/tdunning/t-digest), which allows accurate estimation of quantiles and relative rank over streaming data.

## Features

- **Quantile Estimation**: Compute approximate quantiles (e.g., median, percentiles) from large datasets.
- **Streaming Data**: Suitable for online computation where data arrives incrementally.
- **Merging Digests**: Supports merging t-digests from different data partitions, ideal for distributed systems.
- **Compression**: Adjustable compression factor to balance accuracy and memory usage.
- **Simple State**: Minimal state structure for easy serialization and deserialization.

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
tdigests = 1.0
```

## Usage

```rust
use tdigests::TDigest;

fn main() {
    // Create a new t-digest
    let digest = TDigest::from_values(vec![1.0, 2.0, 3.0]);

    // Estimate quantiles
    let median = digest.estimate_quantile(0.5);
    println!("Estimated median: {}", median);

    // Estimate rank
    let rank = digest.estimate_rank(2.5);
    println!("Rank of 2.5: {}", rank);
}
```

## Examples

See the [`simple.rs`](examples/simple.rs) for a complete example demonstrating how to use the t-digest library.

## Contributing

Contributions are welcome! Please open issues or submit pull requests on the GitHub repository.

## License

This project is licensed under the MIT License.
