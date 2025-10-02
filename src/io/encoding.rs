use anyhow::{Context, Result, anyhow};
use futures_util::future::join_all;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::fs::create_dir_all;
use std::sync::Arc;
use tokio::fs;
use tracing::{info, instrument};

use crate::{
    cli::commands::Commands,
    algorithm::gf256::Gf256,
    codec::{encode_shards::shard_encoding, matrix::build_vandermonde},
};

#[instrument(skip(args))]
pub async fn handle_encode(args: Commands) -> Result<()> {
    let (input_path, out_dir, k, m) = match args {
        Commands::Encode {
            input,
            output,
            data_shards,
            parity_shards,
        } => (input, output, data_shards, parity_shards),
        _ => unreachable!(),
    };

    if k == 0 || m == 0 || k + m > 256 {
        return Err(anyhow!("Invalid k/m values. Must be > 0 and k+m <= 256"));
    }

    let gf = Arc::new(Gf256::new());

    info!("Reading input file: {:?}", input_path);
    let buf = fs::read(&input_path)
        .await
        .with_context(|| format!("Failed to read input file: {:?}", input_path))?;
    let orig_len = buf.len();

    let shard_len = (orig_len + k - 1) / k;
    let mut data_shards = vec![vec![0u8; shard_len]; k];

    let pb_read = ProgressBar::new(orig_len as u64);
    pb_read.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.cyan/black}] Reading input {bytes}/{total_bytes}",
        )
        .unwrap()
        .progress_chars("=> "),
    );

    data_shards
        .par_iter_mut()
        .zip(buf.par_chunks(shard_len))
        .for_each(|(shard, chunk)| {
            shard[..chunk.len()].copy_from_slice(chunk);
            pb_read.inc(chunk.len() as u64);
        });
    pb_read.finish_with_message("Input file loaded!");

    let pb_compute = ProgressBar::new(m as u64);
    pb_compute.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.yellow/black}] Computing parity {pos}/{len}",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    pb_compute.set_position(0);

    let gf_clone = gf.clone();
    let data_shards_clone = data_shards.clone();
    let parities = tokio::task::spawn_blocking(move || {
        let matrix = build_vandermonde(&gf_clone, k, m);
        let parities = shard_encoding(&gf_clone, &matrix, &data_shards_clone, &pb_compute)?;
        pb_compute.finish_with_message("Parity computed!");
        Ok::<_, anyhow::Error>(parities)
    })
    .await??;

    info!(
        "Writing {} data and {} parity shards to {:?}",
        k, m, out_dir
    );
    create_dir_all(&out_dir)
        .with_context(|| format!("Failed to create output directory: {:?}", out_dir))?;

    let pb_write = ProgressBar::new((k + m) as u64);
    pb_write.set_style(
        ProgressStyle::with_template(
            "[{elapsed_precise}] [{bar:40.green/black}] Writing shards {pos}/{len}",
        )
        .unwrap()
        .progress_chars("=> "),
    );

    let mut shards = data_shards;
    shards.extend(parities);

    let mut write_handles = Vec::with_capacity(k + m);
    for (i, shard_data) in shards.into_iter().enumerate() {
        let path = out_dir.join(format!("shard_{:02}.dat", i));
        let pb_clone = pb_write.clone();
        write_handles.push(tokio::spawn(async move {
            fs::write(path, shard_data).await?;
            pb_clone.inc(1);
            Ok::<_, anyhow::Error>(())
        }));
    }

    for handle in join_all(write_handles).await {
        handle??;
    }
    pb_write.finish_with_message("All shards written!");

    let meta = format!("{}\n{} {}\n", orig_len, k, m);
    fs::write(out_dir.join("meta.txt"), meta).await?;

    info!(
        "✅ Successfully encoded '{}' ({} bytes)",
        input_path.display(),
        orig_len
    );
    Ok(())
}
