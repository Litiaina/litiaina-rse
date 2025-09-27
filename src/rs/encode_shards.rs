use anyhow::{anyhow, Result};
use rayon::prelude::*;
use tracing::{debug, instrument};

use crate::gf::gf256::Gf256;

#[instrument(skip_all, fields(k = data_shards.len(), m = matrix.len()))]
pub fn shard_encoding(
    gf: &Gf256,
    matrix: &[Vec<u8>],
    data_shards: &[Vec<u8>],
) -> Result<Vec<Vec<u8>>> {
    let m = matrix.len();
    if m == 0 {
        return Ok(vec![]);
    }
    let k = matrix[0].len();
    if k != data_shards.len() {
        return Err(anyhow!("Matrix columns must match the number of data shards"));
    }
    let shard_len = data_shards.get(0).map_or(0, |v| v.len());
    if data_shards.iter().any(|s| s.len() != shard_len) {
        return Err(anyhow!("All data shards must have the same length"));
    }

    let mut parities = vec![vec![0u8; shard_len]; m];
    debug!("Starting parallel encoding of parity shards.");

    parities
        .par_iter_mut()
        .enumerate()
        .for_each(|(r, parity)| {
            let row = &matrix[r];
            for (c, ds) in data_shards.iter().enumerate().take(k) {
                let coef = row[c];
                if coef == 0 {
                    continue;
                }

                if coef == 1 {
                    for (p_byte, d_byte) in parity.iter_mut().zip(ds.iter()) {
                        *p_byte ^= *d_byte;
                    }
                } else {
                    for i in 0..shard_len {
                        let prod = gf.mul(coef, ds[i]);
                        parity[i] ^= prod;
                    }
                }
            }
        });

    debug!("Finished parallel encoding.");
    Ok(parities)
}
