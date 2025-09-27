use anyhow::{Context, Result, anyhow};
use rayon::prelude::*;
use tracing::{debug, instrument};

use crate::{
    gf::gf256::Gf256,
    rs::{
        encode_shards::shard_encoding,
        matrix::{build_vandermonde, invert_matrix},
    },
};

#[instrument(skip_all, fields(k = k, m = m, total_shards = shards_opt.len()))]
pub fn shard_reconstruction(
    gf: &Gf256,
    k: usize,
    m: usize,
    shards_opt: &mut [Option<Vec<u8>>],
) -> Result<()> {
    let n = k + m;
    assert_eq!(n, shards_opt.len());

    let shard_len = shards_opt
        .iter()
        .find_map(|s| s.as_ref().map(|v| v.len()))
        .ok_or_else(|| anyhow!("No shards available to determine length"))?;

    let present_indices: Vec<usize> = (0..n).filter(|&i| shards_opt[i].is_some()).collect();
    if present_indices.len() < k {
        return Err(anyhow!(
            "Not enough shards to reconstruct: have {}, need {}",
            present_indices.len(),
            k
        ));
    }

    let vandermonde_base = build_vandermonde(k, m);

    let survivors = &present_indices[0..k];
    let mut a = vec![vec![0u8; k]; k];
    for (row_idx, &global_row) in survivors.iter().enumerate() {
        if global_row < k {
            a[row_idx][global_row] = 1;
        } else {
            a[row_idx].copy_from_slice(&vandermonde_base[global_row - k]);
        }
    }

    let a_inv = invert_matrix(gf, &a).context("Failed to invert the reconstruction matrix")?;

    let survivor_data: Vec<&[u8]> = survivors
        .iter()
        .map(|&idx| shards_opt[idx].as_ref().unwrap().as_slice())
        .collect();

    let missing_data_indices: Vec<usize> = (0..k).filter(|&i| shards_opt[i].is_none()).collect();
    let mut recovered_data = vec![vec![0u8; shard_len]; missing_data_indices.len()];

    debug!("Starting parallel reconstruction of missing data shards.");
    recovered_data
        .par_iter_mut()
        .zip(missing_data_indices.par_iter())
        .for_each(|(out_shard, &original_index)| {
            let inv_row = &a_inv[original_index];
            for (j, sdata) in survivor_data.iter().enumerate().take(k) {
                let coef = inv_row[j];
                if coef == 0 {
                    continue;
                }

                if coef == 1 {
                    for (out_byte, &in_byte) in out_shard.iter_mut().zip(sdata.iter()) {
                        *out_byte ^= in_byte;
                    }
                } else {
                    for t in 0..shard_len {
                        out_shard[t] ^= gf.mul(coef, sdata[t]);
                    }
                }
            }
        });
    debug!("Finished parallel data reconstruction.");

    let mut recovered_iter = recovered_data.into_iter();
    for i in 0..k {
        if shards_opt[i].is_none() {
            shards_opt[i] = Some(recovered_iter.next().unwrap());
        }
    }

    let all_data_shards: Vec<Vec<u8>> = (0..k)
        .map(|i| shards_opt[i].as_ref().unwrap().clone())
        .collect();
    debug!("Re-encoding to recover missing parity shards.");
    let parities = shard_encoding(gf, &vandermonde_base, &all_data_shards)?;

    for r in 0..m {
        if shards_opt[k + r].is_none() {
            shards_opt[k + r] = Some(parities[r].clone());
        }
    }

    Ok(())
}
