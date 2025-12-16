//! # T-Digest Rust Library
//!
//! This library provides an implementation of the [t-digest algorithm](https://github.com/tdunning/t-digest) in Rust.
//! The t-digest algorithm is a data structure for accurate on-line accumulation of rank-based
//! statistics such as quantiles and trimmed means.
//!
//! ## Features
//!
//! - Efficient computation of quantiles over streaming data.
//! - Suitable for large datasets and distributed systems.
//! - Supports merging of t-digests from different data partitions.
//!
//! ## Example
//!
//! ```rust
//! use tdigests::TDigest;
//!
//! // Create a t-digest from a list of values
//! let values = vec![10.0, 20.0, 30.0];
//! let mut digest = TDigest::from_values(values);
//!
//! // Compress the centroids with a maximum of 100 centroids
//! digest.compress(100);
//!
//! // Estimate quantiles
//! let median = digest.estimate_quantile(0.5);
//! assert_eq!(median, 20.0);
//!
//! // Compute the relative rank of a value
//! let rank = digest.estimate_rank(25.0);
//! assert_eq!(rank, 0.75);
//! ```

#![deny(missing_docs)]
#![deny(unsafe_code)]

use std::cmp::Ordering;

/// Represents a centroid in the t-digest, which holds a mean value and an associated weight.
///
/// # Examples
///
/// ```rust
/// use tdigests::Centroid;
///
/// let centroid = Centroid::new(10.0, 5.0);
/// assert_eq!(centroid.mean, 10.0);
/// assert_eq!(centroid.weight, 5.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct Centroid {
    /// The mean value of the centroid.
    pub mean: f64,
    /// The weight (number of observations) associated with the centroid.
    pub weight: f64,
}

impl Centroid {
    /// Creates a new centroid with the given mean and weight.
    pub fn new(mean: f64, weight: f64) -> Self {
        Self { mean, weight }
    }

    /// Adds another centroid to this one, updating the mean and weight.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tdigests::Centroid;
    ///
    /// let mut c1 = Centroid::new(10.0, 2.0);
    /// let c2 = Centroid::new(20.0, 3.0);
    /// c1.add(&c2);
    /// assert_eq!(c1.mean, 16.0);
    /// assert_eq!(c1.weight, 5.0);
    /// ```
    pub fn add(&mut self, other: &Centroid) {
        assert!(other.weight > 0.0);
        if self.weight != 0.0 {
            let total_weight = self.weight + other.weight;
            self.mean += other.weight * (other.mean - self.mean) / total_weight;
            self.weight = total_weight;
        } else {
            self.mean = other.mean;
            self.weight = other.weight;
        }
    }
}

/// The main struct representing a t-digest.
///
/// The `TDigest` struct allows efficient computation of quantiles and supports merging from other
/// t-digests.
///
/// # Examples
///
/// ```rust
/// use tdigests::TDigest;
///
/// let values = vec![1.0, 2.0, 3.0];
/// let digest = TDigest::from_values(values);
///
/// let q = digest.estimate_quantile(0.5);
/// assert_eq!(q, 2.0);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TDigest {
    centroids: Vec<Centroid>,
}

impl TDigest {
    /// Creates a t-digest from a list of values.
    ///
    /// # Panics
    ///
    /// Panics if the input values are empty or if all values are `NaN`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tdigests::TDigest;
    ///
    /// let values = vec![3.0, 1.0, 2.0];
    /// let digest = TDigest::from_values(values);
    /// ```
    pub fn from_values(mut values: Vec<f64>) -> Self {
        values.retain(|v| !v.is_nan());

        assert!(!values.is_empty());

        if values.len() == 1 {
            return Self::from_centroids(vec![Centroid::new(values[0], 1.0)]);
        }

        let min = values
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let max = values
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        let mut met_min = false;
        let mut met_max = false;
        let mut centroids = Vec::with_capacity(values.len());

        centroids.push(Centroid::new(*min, 1.0));

        for value in &values {
            if value == min && !met_min {
                met_min = true;
            } else if value == max && !met_max {
                met_max = true;
            } else {
                centroids.push(Centroid::new(*value, 1.0));
            }
        }

        centroids.push(Centroid::new(*max, 1.0));

        Self::from_centroids(centroids)
    }

    /// Creates a t-digest from existing centroids.
    ///
    /// This method is useful when working in distributed systems where centroids
    /// are generated on partitions of data and need to be merged.
    ///
    /// # Panics
    ///
    /// Panics if the input centroids is empty or if all centroids have either zero weight or `NaN`
    /// means.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tdigests::Centroid;
    /// use tdigests::TDigest;
    ///
    /// let centroids = vec![
    ///     Centroid::new(10.0, 1.0),
    ///     Centroid::new(20.0, 2.0),
    ///     Centroid::new(30.0, 1.0),
    /// ];
    /// let digest = TDigest::from_centroids(centroids);
    /// ```
    pub fn from_centroids(centroids: Vec<Centroid>) -> Self {
        let mut tdigest = Self { centroids };

        tdigest
            .centroids
            .retain(|c| c.weight > 0.0 && !c.mean.is_nan());
        assert!(!tdigest.centroids.is_empty());

        // Sort centroids by mean
        tdigest
            .centroids
            .sort_by(|a, b| a.mean.partial_cmp(&b.mean).unwrap());

        tdigest
    }

    /// Returns the internal centroids.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tdigests::TDigest;
    ///
    /// let values = vec![1.0, 2.0, 3.0];
    /// let digest = TDigest::from_values(values);
    ///
    /// let centroids = digest.centroids();
    /// assert_eq!(centroids.len(), 3);
    /// ```
    pub fn centroids(&self) -> &[Centroid] {
        &self.centroids
    }

    /// Merges another t-digest into this one.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tdigests::TDigest;
    ///
    /// let values1 = vec![1.0, 2.0, 3.0];
    /// let digest1 = TDigest::from_values(values1);
    ///
    /// let values2 = vec![4.0, 5.0, 6.0];
    /// let digest2 = TDigest::from_values(values2);
    ///
    /// let merged_digest = digest1.merge(&digest2);
    /// ```
    pub fn merge(&self, other: &TDigest) -> TDigest {
        TDigest::from_centroids([self.centroids(), other.centroids()].concat())
    }

    /// Compresses the centroids to reduce their number while maintaining accuracy.
    ///
    /// This method should be called after adding a large number of points to reduce
    /// the memory footprint and improve estimation performance.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tdigests::TDigest;
    ///
    /// let values = (0..1000).map(|i| i as f64).collect::<Vec<_>>();
    /// let mut digest = TDigest::from_values(values);
    ///
    /// // Compress to have at most 100 centroids
    /// digest.compress(100);
    ///
    /// assert!(digest.centroids().len() <= 100);
    /// ```
    pub fn compress(&mut self, max_centroids: usize) {
        let max_centroids = max_centroids.max(3);

        if self.centroids.len() <= max_centroids {
            return;
        }

        // Take min and max centroids
        let first_centroid = self.centroids[0].clone();
        let last_centroid = self.centroids[self.centroids.len() - 1].clone();
        let min = if first_centroid.weight <= 1.0 {
            self.centroids.remove(0)
        } else {
            self.centroids.first_mut().unwrap().weight -= 1.0;
            Centroid::new(first_centroid.mean, 1.0)
        };
        let max = if last_centroid.weight <= 1.0 {
            self.centroids.pop().unwrap()
        } else {
            self.centroids.last_mut().unwrap().weight -= 1.0;
            Centroid::new(last_centroid.mean, 1.0)
        };

        let mut merged = Vec::with_capacity(max_centroids);

        // Put min centroids back
        merged.push(min);

        // Min and max centroids take up two slots
        let max_centroids = max_centroids - 2;
        let total_weight = self.total_weight();
        let mut k_limit = 1;
        let mut q_limit = Self::k_to_q(k_limit as f64 / max_centroids as f64);
        let mut weight_so_far = 0.0;
        let mut current = self.centroids[0].clone();
        weight_so_far += current.weight;

        for centroid in self.centroids.iter().skip(1) {
            let proposed_weight = weight_so_far + centroid.weight;
            let q = proposed_weight / total_weight;
            if q <= q_limit || k_limit == max_centroids {
                current.add(centroid);
                weight_so_far = proposed_weight;
            } else {
                merged.push(current);
                current = centroid.clone();
                weight_so_far = proposed_weight;
                k_limit += 1;
                q_limit = Self::k_to_q(k_limit as f64 / max_centroids as f64);
            }
        }
        merged.push(current);

        // Put max centroids back
        merged.push(max);

        self.centroids = merged;
    }

    /// Converts a centroid index, which has been scaled to [0, 1], into a quantile.
    fn k_to_q(k_ratio: f64) -> f64 {
        if k_ratio >= 0.5 {
            let base = 1.0 - k_ratio;
            1.0 - 2.0 * base * base
        } else {
            2.0 * k_ratio * k_ratio
        }
    }

    /// Estimates the quantile for a given cumulative probability `q`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tdigests::TDigest;
    ///
    /// let values = (0..=100).map(|i| i as f64).collect::<Vec<_>>();
    /// let digest = TDigest::from_values(values);
    /// let median = digest.estimate_quantile(0.5);
    /// assert_eq!(median, 50.0);
    /// ```
    pub fn estimate_quantile(&self, q: f64) -> f64 {
        let q = q.clamp(0.0, 1.0);

        if self.centroids.len() == 1 {
            return self.centroids[0].mean;
        }

        let total_weight = self.total_weight();
        let mut cumulative = 0.0;
        let mut cum_left = 0.0;
        let mut cum_right = 0.0;
        let mut position = 0;

        for (k, centroid) in self.centroids.iter().enumerate() {
            cum_left = cum_right;
            cum_right = (2.0 * cumulative + centroid.weight - 1.0) / 2.0 / (total_weight - 1.0);
            cumulative += centroid.weight;

            if cum_right >= q {
                break;
            }

            position = k + 1;
        }

        if position == 0 {
            return self.centroids[0].mean;
        }

        if position >= self.centroids.len() {
            return self.centroids[self.centroids.len() - 1].mean;
        }

        let centroid_left = &self.centroids[position - 1];
        let centroid_right = &self.centroids[position];

        let weight_between = cum_right - cum_left;
        let fraction = (q - cum_left) / weight_between;

        centroid_left.mean * (1.0 - fraction) + centroid_right.mean * fraction
    }

    /// Compute the relative rank of a given value `x`, specified as a percentage ranging from 0.0
    /// to 1.0.
    ///
    /// Returns a value between 0.0 and 1.0 representing the probability that a random
    /// variable is less than or equal to `x`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use tdigests::TDigest;
    ///
    /// let values = (0..=100).map(|i| i as f64).collect::<Vec<_>>();
    /// let digest = TDigest::from_values(values);
    /// let rank = digest.estimate_rank(50.0);
    /// assert_eq!(rank, 0.5);
    /// ```
    pub fn estimate_rank(&self, x: f64) -> f64 {
        if self.centroids.len() == 1 {
            return match self.centroids[0].mean.partial_cmp(&x).unwrap() {
                Ordering::Less => 1.0,
                Ordering::Equal => 0.5,
                Ordering::Greater => 0.0,
            };
        }

        let total_weight = self.total_weight();
        let mut cumulative = 0.0;
        let mut cum_left = 0.0;
        let mut cum_right = 0.0;
        let mut position = 0;

        for (k, centroid) in self.centroids.iter().enumerate() {
            cum_left = cum_right;
            cum_right = (2.0 * cumulative + centroid.weight - 1.0) / 2.0 / (total_weight - 1.0);
            cumulative += centroid.weight;

            if centroid.mean >= x {
                break;
            }

            position = k + 1;
        }

        if position == 0 {
            return 0.0;
        }

        if position >= self.centroids.len() {
            return 1.0;
        }

        let centroid_left = &self.centroids[position - 1];
        let centroid_right = &self.centroids[position];

        let weight_between = cum_right - cum_left;
        let fraction = (x - centroid_left.mean) / (centroid_right.mean - centroid_left.mean);

        cum_left + fraction * weight_between
    }

    /// Returns the total weight of all centroids.
    fn total_weight(&self) -> f64 {
        let total_weight = self.centroids.iter().map(|c| c.weight).sum();
        assert_ne!(total_weight, 0.0);
        total_weight
    }
}

/// Computes the empirical quantile from data.
#[doc(hidden)]
pub fn naive_quantile(data: &[f64], q: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let pos = (n as f64 - 1.0) * q;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = pos - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}

/// Computes the empirical rank from data.
#[doc(hidden)]
pub fn naive_rank(data: &[f64], x: f64) -> f64 {
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pos = sorted.iter().filter(|&&v| v <= x).count();
    if pos == 0 {
        return 0.0;
    }
    if pos == sorted.len() {
        return 1.0;
    }
    let lower = pos - 1;
    let upper = pos;
    if lower == upper {
        pos as f64 / sorted.len() as f64
    } else {
        let fraction = (x - sorted[lower]) / (sorted[upper] - sorted[lower]);
        let lower_q = lower as f64 / (sorted.len() - 1) as f64;
        let upper_q = upper as f64 / (sorted.len() - 1) as f64;
        lower_q + fraction * (upper_q - lower_q)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;
    use rand::distributions::Distribution;
    use rand::distributions::Uniform;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    use super::*;

    #[test]
    fn test_single_value() {
        let values = vec![42.0];
        let mut digest = TDigest::from_values(values.clone());
        digest.compress(100);
        for q in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            assert_eq!(digest.estimate_quantile(*q), 42.0);
        }
    }

    #[test]
    fn test_few_values() {
        let values = vec![10.0, 20.0, 30.0];
        let digest = TDigest::from_values(values.clone());
        let quantiles = [0.0, 0.25, 0.5, 0.75, 1.0];
        for &q in &quantiles {
            let expected = naive_quantile(&values, q);
            let estimated = digest.estimate_quantile(q);
            assert!(
                (expected - estimated).abs() <= 1e-6,
                "q={}, expected={}, estimated={}",
                q,
                expected,
                estimated
            );
        }
    }

    #[test]
    fn test_min_max() {
        let values = vec![10.0, 10.0, 20.0, 30.0, 30.0];
        let digest = TDigest::from_values(values.clone());
        let quantiles = [0.0, 0.25, 0.5, 0.75, 1.0];
        for &q in &quantiles {
            let expected = naive_quantile(&values, q);
            let estimated = digest.estimate_quantile(q);
            assert!(
                (expected - estimated).abs() <= 1e-6,
                "q={}, expected={}, estimated={}",
                q,
                expected,
                estimated
            );
        }
    }

    #[test]
    fn test_large_dataset() {
        let mut rng = StdRng::seed_from_u64(42);
        let uniform = Uniform::new(0.0, 1.0);
        let values = (0..10000)
            .map(|_| uniform.sample(&mut rng))
            .collect::<Vec<_>>();
        let mut digest = TDigest::from_values(values.clone());
        digest.compress(100);
        let quantiles = [0.0, 0.25, 0.5, 0.75, 1.0];
        for &q in &quantiles {
            let expected = naive_quantile(&values, q);
            let estimated = digest.estimate_quantile(q);
            assert!(
                (expected - estimated).abs() <= 0.01,
                "q={}, expected={}, estimated={}",
                q,
                expected,
                estimated
            );
        }
    }

    #[test]
    fn test_merge() {
        let mut rng = StdRng::seed_from_u64(42);
        let uniform = Uniform::new(0.0, 1.0);
        let values1 = (0..5000)
            .map(|_| uniform.sample(&mut rng))
            .collect::<Vec<_>>();
        let values2 = (0..5000)
            .map(|_| uniform.sample(&mut rng))
            .collect::<Vec<_>>();
        let mut values = Vec::new();
        values.extend(&values1);
        values.extend(&values2);

        let mut digest1 = TDigest::from_values(values1.clone());
        let mut digest2 = TDigest::from_values(values2.clone());
        digest1.compress(100);
        digest2.compress(100);

        let merged_digest = digest1.merge(&digest2);

        let quantiles = [0.0, 0.25, 0.5, 0.75, 1.0];
        for &q in &quantiles {
            let expected = naive_quantile(&values, q);
            let estimated = merged_digest.estimate_quantile(q);
            assert!(
                (expected - estimated).abs() <= 0.01,
                "q={}, expected={}, estimated={}",
                q,
                expected,
                estimated
            );
        }
    }

    #[test]
    fn test_monotonicity() {
        let mut rng = StdRng::seed_from_u64(42);
        let uniform = Uniform::new(0.0, 1.0);
        let values: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
        let mut digest = TDigest::from_values(values.clone());
        digest.compress(100);
        let mut last_quantile = -1.0;
        for q in (0..1000).map(|i| i as f64 / 1000.0) {
            let quantile = digest.estimate_quantile(q);
            assert!(
                quantile >= last_quantile,
                "q={}, quantile={}, last_quantile={}",
                q,
                quantile,
                last_quantile
            );
            last_quantile = quantile;
        }
    }

    #[test]
    fn test_rank() {
        let mut rng = StdRng::seed_from_u64(42);
        let uniform = Uniform::new(0.0, 1.0);
        let values: Vec<f64> = (0..10000).map(|_| uniform.sample(&mut rng)).collect();
        let mut digest = TDigest::from_values(values.clone());
        digest.compress(100);
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        for &x in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let estimated = digest.estimate_rank(x);
            let expected = naive_rank(&sorted_values, x);
            assert!(
                (estimated - expected).abs() <= 0.01,
                "x={}, estimated={}, expected={}",
                x,
                estimated,
                expected
            );
        }
    }

    #[test]
    fn test_construct_from_centroids() {
        let mut rng = StdRng::seed_from_u64(12345);
        let uniform = Uniform::new(0.0, 100.0);
        let values: Vec<f64> = (0..1000).map(|_| uniform.sample(&mut rng)).collect();
        let digest = TDigest::from_values(values.clone());

        // Construct a new digest from the centroids
        let centroids = digest.centroids().to_vec();
        let digest_from_centroids = TDigest::from_centroids(centroids);

        // The two digests should be equal
        assert_eq!(digest, digest_from_centroids);

        // Reconstruct data points for testing
        let values: Vec<f64> = digest
            .centroids
            .iter()
            .flat_map(|c| std::iter::repeat(c.mean).take(c.weight.round() as usize))
            .collect();

        // Test quantile estimation
        let quantiles = [0.0, 0.25, 0.5, 0.75, 1.0];
        for &q in &quantiles {
            let estimated_original = digest.estimate_quantile(q);
            let estimated_from_centroids = digest_from_centroids.estimate_quantile(q);
            let expected = naive_quantile(&values, q);
            assert!(
                (estimated_original - estimated_from_centroids).abs() <= 1e-6,
                "q={}, original={}, from_centroids={}",
                q,
                estimated_original,
                estimated_from_centroids
            );
            assert!(
                (estimated_original - expected).abs() <= 1e-6,
                "q={}, estimated={}, expected={}",
                q,
                estimated_original,
                expected
            );
        }

        // Test rank estimation
        let test_values = [0.0, 25.0, 50.0, 75.0, 100.0];
        for &x in &test_values {
            let estimated_original = digest.estimate_rank(x);
            let estimated_from_centroids = digest_from_centroids.estimate_rank(x);
            let expected = naive_rank(&values, x);
            assert!(
                (estimated_original - estimated_from_centroids).abs() <= 1e-6,
                "x={}, original_rank={}, from_centroids_rank={}",
                x,
                estimated_original,
                estimated_from_centroids
            );
            assert!(
                (estimated_original - expected).abs() <= 0.02,
                "x={}, estimated_rank={}, expected_rank={}",
                x,
                estimated_original,
                expected
            );
        }
    }

    #[test]
    fn test_distributed_merge() {
        // Simulate distributed computation
        let mut rng = StdRng::seed_from_u64(42);
        let uniform = Uniform::new(0.0, 100.0);
        let values1: Vec<f64> = (0..1000).map(|_| uniform.sample(&mut rng)).collect();
        let values2: Vec<f64> = (0..1000).map(|_| uniform.sample(&mut rng)).collect();
        let values3: Vec<f64> = (0..1000).map(|_| uniform.sample(&mut rng)).collect();
        let mut values: Vec<f64> = Vec::new();
        values.extend(&values1);
        values.extend(&values2);
        values.extend(&values3);

        let mut digest1 = TDigest::from_values(values1.clone());
        let mut digest2 = TDigest::from_values(values2.clone());
        let mut digest3 = TDigest::from_values(values3.clone());
        digest1.compress(100);
        digest2.compress(100);
        digest3.compress(100);

        // Merge digests by merging centroids
        let merged_digest = digest1.merge(&digest2).merge(&digest3);

        // Alternatively, merge digests directly
        let merged_digest_direct = TDigest::from_centroids(
            [
                digest1.centroids(),
                digest2.centroids(),
                digest3.centroids(),
            ]
            .concat(),
        );

        // The two merged digests should be approximately equal
        assert!((merged_digest.total_weight() - merged_digest_direct.total_weight()).abs() < 1e-6);

        // Sort values for comparison
        values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Test quantile estimation
        let quantiles = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0];
        for &q in &quantiles {
            let expected = naive_quantile(&values, q);
            let estimated = merged_digest.estimate_quantile(q);
            let estimated_direct = merged_digest_direct.estimate_quantile(q);
            assert!(
                (estimated - expected).abs() <= 1.0,
                "q={}, estimated={}, expected={}",
                q,
                estimated,
                expected
            );
            assert!(
                (estimated - estimated_direct).abs() <= 1e-6,
                "q={}, estimated={}, estimated_direct={}",
                q,
                estimated,
                estimated_direct
            );
        }

        // Test rank estimation
        let test_values = [0.0, 10.0, 25.0, 50.0, 75.0, 90.0, 100.0];
        for &x in &test_values {
            let expected_rank = naive_rank(&values, x);
            let estimated_rank = merged_digest.estimate_rank(x);
            let estimated_rank_direct = merged_digest_direct.estimate_rank(x);
            assert!(
                (estimated_rank - expected_rank).abs() <= 0.01,
                "x={}, estimated_rank={}, expected_rank={}",
                x,
                estimated_rank,
                expected_rank
            );
            assert!(
                (estimated_rank - estimated_rank_direct).abs() <= 0.01,
                "x={}, estimated_rank={}, estimated_rank_direct={}",
                x,
                estimated_rank,
                estimated_rank_direct
            );
        }
    }

    #[test]
    fn test_max_centroids() {
        let values: Vec<f64> = (1..=10000).map(|i| i as f64).collect();

        let mut digest = TDigest::from_values(values.clone());
        digest.compress(100);
        assert_eq!(digest.centroids().len(), 100);
        digest.compress(10);
        assert_eq!(digest.centroids().len(), 10);
    }
}
