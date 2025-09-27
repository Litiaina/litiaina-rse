//! # Litiaina Reed Solomon Erasure GF(2^&)
//!
//! ## Usage
//!
//! ### Encoding a file
//!
//! ```bash
//! RUST_LOG=info cargo run --release -- encode --input my_large_file.bin --output shards_out --data-shards 10 --parity-shards 4
//! ```
//!
//! ### Reconstructing a file
//!
//! ```bash
//! RUST_LOG=info cargo run --release -- decode --input shards_out --output recovered_file.bin
//! ```

mod cli;
mod gf;
mod io;
mod rs;

use anyhow::Result;
use clap::Parser;
use std::sync::Arc;
use std::time::Instant;
use tracing::{Level, error, info};
use tracing_subscriber::FmtSubscriber;

use crate::{
    cli::commands::{Cli, Commands},
    gf::gf256::Gf256,
    io::{decoding::handle_decode, encoding::handle_encode},
};

#[cfg(test)]
mod tests {
    use crate::{
        gf::gf256::Gf256,
        rs::{
            encode_shards::shard_encoding,
            matrix::{build_vandermonde, invert_matrix},
            reconstruct_shards::shard_reconstruction,
        },
    };

    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() -> Result<()> {
        let gf = Gf256::new();
        let k = 10;
        let m = 4;
        let shard_len = 8192;

        let data_shards: Vec<Vec<u8>> = (0..k)
            .map(|i| {
                (0..shard_len)
                    .map(|j| ((i + 1) as u8).wrapping_mul(j as u8))
                    .collect()
            })
            .collect();

        let parities = shard_encoding(&gf, &build_vandermonde(k, m), &data_shards)?;
        assert_eq!(parities.len(), m);

        let n = k + m;
        let mut shards_opt: Vec<Option<Vec<u8>>> = (0..n).map(|_| None).collect();
        for i in 0..k {
            shards_opt[i] = Some(data_shards[i].clone());
        }
        for r in 0..m {
            shards_opt[k + r] = Some(parities[r].clone());
        }

        shards_opt[1] = None;
        shards_opt[3] = None;
        shards_opt[k] = None;
        shards_opt[k + 2] = None;

        shard_reconstruction(&gf, k, m, &mut shards_opt)?;

        for i in 0..k {
            let original = &data_shards[i];
            let reconstructed = shards_opt[i].as_ref().unwrap();
            assert_eq!(
                original, reconstructed,
                "Shard {} was not reconstructed correctly",
                i
            );
        }
        Ok(())
    }

    #[test]
    fn test_singular_matrix_inversion() {
        let gf = Gf256::new();
        let singular_matrix = vec![vec![1, 1], vec![2, 2]];
        let result = invert_matrix(&gf, &singular_matrix);
        assert!(result.is_err());
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::TRACE)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .finish();
    tracing::subscriber::set_global_default(subscriber).expect("setting default subscriber failed");

    let cli = Cli::parse();

    let gf = Arc::new(Gf256::new());

    let start_time = Instant::now();

    let result = match cli.command {
        Commands::Encode { .. } => handle_encode(gf.clone(), cli.command).await,
        Commands::Decode { .. } => handle_decode(gf.clone(), cli.command).await,
    };

    if let Err(e) = &result {
        error!("Operation failed: {:?}", e);
    }

    info!("Total execution time: {:.2?}", start_time.elapsed());

    result
}
