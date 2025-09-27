use anyhow::{anyhow, Context, Result};
use futures_util::future::join_all;
use rayon::prelude::*;
use std::fs::create_dir_all;
use std::sync::Arc;
use tokio::fs;
use tracing::{info, instrument};

use crate::{
    cli::commands::Commands,
    gf::gf256::Gf256,
    rs::{encode_shards::shard_encoding, matrix::build_vandermonde},
};

#[instrument(skip(gf, args))]
pub async fn handle_encode(gf: Arc<Gf256>, args: Commands) -> Result<()> {
    let (input_path, out_dir, k, m) = match args {
        Commands::Encode {
            input,
            output,
            data_shards,
            parity_shards,
        } => (input, output, data_shards, parity_shards),
        _ => unreachable!(),
    };

    if k == 0 || m == 0 || k + m > 255 {
        return Err(anyhow!("Invalid k/m values. Must be > 0 and k+m <= 255"));
    }

    info!("Reading input file: {:?}", input_path);
    let buf = fs::read(&input_path)
        .await
        .with_context(|| format!("Failed to read input file: {:?}", input_path))?;
    let orig_len = buf.len();

    let shard_len = (orig_len + k - 1) / k;
    let mut data_shards = vec![vec![0u8; shard_len]; k];

    data_shards
        .par_iter_mut()
        .zip(buf.par_chunks(shard_len))
        .for_each(|(shard, chunk)| {
            shard[..chunk.len()].copy_from_slice(chunk);
        });

    let gf_clone = gf.clone();
    let data_shards_clone = data_shards.clone();
    let parities = tokio::task::spawn_blocking(move || {
        let matrix = build_vandermonde(k, m);
        shard_encoding(&gf_clone, &matrix, &data_shards_clone)
    })
    .await??;

    info!(
        "Writing {} data and {} parity shards to {:?}",
        k, m, out_dir
    );
    create_dir_all(&out_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", out_dir))?;

    let mut write_handles = Vec::with_capacity(k + m);

    for i in 0..k {
        let path = out_dir.join(format!("data_{}.shard", i));
        let data = data_shards[i].clone();
        write_handles.push(tokio::spawn(
            async move { fs::write(path, data).await },
        ));
    }

    for r in 0..m {
        let path = out_dir.join(format!("parity_{}.shard", r));
        let data = parities[r].clone();
        write_handles.push(tokio::spawn(
            async move { fs::write(path, data).await },
        ));
    }

    for handle in join_all(write_handles).await {
        handle?.context("A shard write operation failed")?;
    }

    let meta = format!("{}\n{} {}\n", orig_len, k, m);
    fs::write(out_dir.join("meta.txt"), meta).await?;

    info!(
        "✅ Successfully encoded '{}' ({} bytes)",
        input_path.display(),
        orig_len
    );
    Ok(())
}
