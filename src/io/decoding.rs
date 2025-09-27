use anyhow::{Context, Result, anyhow};
use futures_util::future::join_all;
use std::sync::Arc;
use tokio::fs;
use tracing::{info, instrument, warn};

use crate::{
    cli::commands::Commands, gf::gf256::Gf256, rs::reconstruct_shards::shard_reconstruction,
};

#[instrument(skip(gf, args))]
pub async fn handle_decode(gf: Arc<Gf256>, args: Commands) -> Result<()> {
    let (shard_dir, output_path) = match args {
        Commands::Decode { input, output } => (input, output),
        _ => unreachable!(),
    };

    info!("Reading metadata from: {:?}", shard_dir);
    let meta_raw = fs::read_to_string(shard_dir.join("meta.txt"))
        .await
        .context("Failed to read meta.txt. Is the shard directory correct?")?;
    let mut lines = meta_raw.lines();
    let orig_len: usize = lines
        .next()
        .ok_or(anyhow!("Invalid meta.txt: missing length"))?
        .trim()
        .parse()?;
    let km_line = lines
        .next()
        .ok_or(anyhow!("Invalid meta.txt: missing k/m"))?;
    let mut parts = km_line.split_whitespace();
    let k: usize = parts
        .next()
        .ok_or(anyhow!("Invalid meta.txt: missing k"))?
        .parse()?;
    let m: usize = parts
        .next()
        .ok_or(anyhow!("Invalid meta.txt: missing m"))?
        .parse()?;

    let n = k + m;
    let mut shards_opt: Vec<Option<Vec<u8>>> = vec![None; n];

    info!("Concurrently reading available shards...");
    let mut read_handles = Vec::with_capacity(n);
    for i in 0..n {
        let path = if i < k {
            shard_dir.join(format!("data_{}.shard", i))
        } else {
            shard_dir.join(format!("parity_{}.shard", i - k))
        };
        read_handles.push(tokio::spawn(async move {
            if path.exists() {
                fs::read(&path).await.ok()
            } else {
                warn!("Shard missing: {}", path.display());
                None
            }
        }));
    }

    let results = join_all(read_handles).await;
    for (i, result) in results.into_iter().enumerate() {
        shards_opt[i] = result.context("A shard read operation failed")?;
    }

    let missing_count = shards_opt.iter().filter(|s| s.is_none()).count();
    let final_shards = if missing_count > 0 {
        info!(
            "Found {} missing shards. Attempting reconstruction...",
            missing_count
        );

        let reconstructed_shards = tokio::task::spawn_blocking(move || {
            shard_reconstruction(&gf, k, m, &mut shards_opt)?;
            Ok::<_, anyhow::Error>(shards_opt)
        })
        .await??;
        reconstructed_shards
    } else {
        info!("All shards present, no reconstruction needed.");
        shards_opt
    };

    info!("Assembling final file: {:?}", output_path);
    let shard_len = (orig_len + k - 1) / k;
    let mut out_buf = Vec::with_capacity(shard_len * k);
    for i in 0..k {
        let shard = final_shards[i]
            .as_ref()
            .context("Reconstructed data shard is missing unexpectedly")?;
        out_buf.extend_from_slice(shard);
    }
    out_buf.truncate(orig_len);

    fs::write(&output_path, &out_buf).await?;

    info!(
        "✅ Successfully reconstructed '{}' ({} bytes)",
        output_path.display(),
        orig_len
    );
    Ok(())
}
