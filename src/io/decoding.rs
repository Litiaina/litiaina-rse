use anyhow::{Context, Result, anyhow};
use futures_util::future::join_all;
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::Arc;
use tokio::fs;
use tracing::{info, instrument};

use crate::{cli::commands::Commands, codec::reconstruct_shards::Codec};

#[instrument(skip(args))]
pub async fn handle_decode(args: Commands) -> Result<()> {
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

    let codec = Arc::new(Codec::new(k, m));

    let n = k + m;
    let mut shards_opt: Vec<Option<Vec<u8>>> = vec![None; n];

    info!("Reading available shards...");
    let pb = ProgressBar::new(n as u64);
    pb.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.green/black}] Reading shards {pos}/{len}",
        )
        .unwrap()
        .progress_chars("=> "),
    );

    let mut read_handles = Vec::with_capacity(n);
    for i in 0..n {
        let path = shard_dir.join(format!("shard_{:02}.dat", i));
        let pb_clone = pb.clone();
        read_handles.push(tokio::spawn(async move {
            let data = if path.exists() {
                Some(fs::read(&path).await?)
            } else {
                None
            };
            pb_clone.inc(1);
            Ok::<Option<Vec<u8>>, anyhow::Error>(data)
        }));
    }

    let results = join_all(read_handles).await;
    for (i, result) in results.into_iter().enumerate() {
        let shard = result.context("Join error in shard read task")??;
        shards_opt[i] = shard;
    }
    pb.finish_with_message("Shards read!");

    let missing_count = shards_opt.iter().filter(|s| s.is_none()).count();

    if missing_count > 0 {
        info!("Found {} missing shards. Reconstructing...", missing_count);

        let pb_recon = ProgressBar::new(missing_count as u64);
        pb_recon.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] [{bar:40.yellow/black}] Reconstructing {pos}/{len}",
            )
            .unwrap()
            .progress_chars("=> "),
        );

        let codec_clone = codec.clone();
        shards_opt =
            tokio::task::spawn_blocking(move || -> Result<Vec<Option<Vec<u8>>>, anyhow::Error> {
                let mut shards_to_reconstruct = shards_opt;
                codec_clone.reconstruct(&mut shards_to_reconstruct)?;
                pb_recon.finish_with_message("Reconstruction complete!");
                Ok(shards_to_reconstruct)
            })
            .await
            .context("Shard reconstruction task panicked")??;
    } else {
        info!("All shards present, no reconstruction needed.");
    };

    info!("Assembling final file: {:?}", output_path);
    let pb_write = ProgressBar::new(orig_len as u64);
    pb_write.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.magenta/black}] Writing output {bytes}/{total_bytes}",
        )
        .unwrap()
        .progress_chars("=> "),
    );

    let mut out_buf = Vec::with_capacity(orig_len);
    let mut bytes_written = 0;
    for i in 0..k {
        let shard = shards_opt[i]
            .as_ref()
            .context("Reconstructed data shard is missing unexpectedly")?;
        let to_write = std::cmp::min(shard.len(), orig_len - bytes_written);
        out_buf.extend_from_slice(&shard[..to_write]);
        bytes_written += to_write;
        pb_write.inc(to_write as u64);
    }
    pb_write.finish_with_message("File assembled!");

    fs::write(&output_path, &out_buf).await?;

    info!(
        "✅ Successfully reconstructed '{}' ({} bytes)",
        output_path.display(),
        orig_len
    );
    Ok(())
}
