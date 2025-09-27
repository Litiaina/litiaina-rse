use anyhow::{Context, Result, anyhow};
use futures_util::future::join_all;
use std::sync::Arc;
use tokio::fs;
use tracing::{info, instrument};
use indicatif::{ProgressBar, ProgressStyle};

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

    info!("Reading available shards...");
    let pb = ProgressBar::new(n as u64);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] [{bar:40.green/black}] Reading shards {pos}/{len}")
            .unwrap()
            .progress_chars("=> "),
    );

    let mut read_handles = Vec::with_capacity(n);
    for i in 0..n {
        let path = if i < k {
            shard_dir.join(format!("data_{}.shard", i))
        } else {
            shard_dir.join(format!("parity_{}.shard", i - k))
        };
        let pb = pb.clone();
        read_handles.push(tokio::spawn(async move {
            let result = if path.exists() { fs::read(&path).await.ok() } else { None };
            pb.inc(1);
            result
        }));
    }

    let results = join_all(read_handles).await;
    for (i, result) in results.into_iter().enumerate() {
        shards_opt[i] = result.context("A shard read operation failed")?;
    }
    pb.finish_with_message("Shards read!");

    let missing_count = shards_opt.iter().filter(|s| s.is_none()).count();
    let final_shards = if missing_count > 0 {
        info!("Found {} missing shards. Reconstructing...", missing_count);

        let pb = ProgressBar::new(missing_count as u64);
        pb.set_style(
            ProgressStyle::with_template("[{elapsed_precise}] [{bar:40.yellow/black}] Reconstructing {pos}/{len}")
                .unwrap()
                .progress_chars("=> "),
        );

        let gf_clone = gf.clone();
        let mut shards_clone = shards_opt.clone();

        let reconstructed_shards = tokio::task::spawn_blocking(move || {
            let missing_indices: Vec<usize> = shards_clone.iter().enumerate()
                .filter(|(_, s)| s.is_none())
                .map(|(i, _)| i)
                .collect();

            shard_reconstruction(&gf_clone, k, m, &mut shards_clone)?;

            for _ in missing_indices {
                pb.inc(1);
            }
            pb.finish_with_message("Reconstruction complete!");
            Ok::<_, anyhow::Error>(shards_clone)
        }).await??;

        reconstructed_shards
    } else {
        info!("All shards present, no reconstruction needed.");
        shards_opt
    };

    info!("Assembling final file: {:?}", output_path);
    let pb = ProgressBar::new(orig_len as u64);
    pb.set_style(
        ProgressStyle::with_template("[{elapsed_precise}] [{bar:40.magenta/black}] Writing output {bytes}/{total_bytes}")
            .unwrap()
            .progress_chars("=> "),
    );

    let shard_len = (orig_len + k - 1) / k;
    let mut out_buf = Vec::with_capacity(shard_len * k);
    for i in 0..k {
        let shard = final_shards[i]
            .as_ref()
            .context("Reconstructed data shard is missing unexpectedly")?;
        for byte in shard {
            out_buf.push(*byte);
            pb.inc(1);
        }
    }
    pb.finish_with_message("File assembled!");
    out_buf.truncate(orig_len);

    fs::write(&output_path, &out_buf).await?;

    info!("✅ Successfully reconstructed '{}' ({} bytes)", output_path.display(), orig_len);
    Ok(())
}
