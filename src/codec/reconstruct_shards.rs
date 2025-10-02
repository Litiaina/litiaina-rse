use crate::{
    algorithm::gf256::Gf256,
    codec::matrix::{Matrix, build_vandermonde, invert_matrix, mul_vec_matrix},
};
use anyhow::{Context, Result, anyhow};
use dashmap::DashMap;
use rayon::prelude::*;
use tracing::{info_span, instrument};

pub struct Codec {
    k: usize,
    m: usize,
    n: usize,
    gf: Gf256,
    /// Encoding matrix, also used for reconstruction.
    /// This is a Vandermonde matrix of size m x k.
    encode_matrix: Matrix,
    /// Cache for inverted matrices, keyed by the sorted indices of survivor shards.
    inverse_matrix_cache: DashMap<Vec<usize>, Matrix>,
}

impl Codec {
    pub fn new(k: usize, m: usize) -> Self {
        let gf = Gf256::new();
        let encode_matrix = build_vandermonde(&gf, k, m);
        Self {
            k,
            m,
            n: k + m,
            gf,
            encode_matrix,
            inverse_matrix_cache: DashMap::new(),
        }
    }

    fn get_or_compute_inverse_matrix(&self, survivors: &[usize]) -> Result<Matrix> {
        let mut key = survivors.to_vec();
        key.sort_unstable();

        if let Some(cached_inv) = self.inverse_matrix_cache.get(&key) {
            return Ok(cached_inv.value().clone());
        }

        // Build the k x k matrix `A` from the survivor shards.
        // The rows of `A` are the rows of the original encoding matrix.
        // If the survivor is a data shard i < k, the row is an identity row.
        // If the survivor is a parity shard i >= k, the row is from the Vandermonde matrix.
        let mut a = vec![vec![0u8; self.k]; self.k];
        for (row_idx, &global_row_idx) in survivors.iter().enumerate() {
            if global_row_idx < self.k {
                a[row_idx][global_row_idx] = 1;
            } else {
                a[row_idx].copy_from_slice(&self.encode_matrix[global_row_idx - self.k]);
            }
        }

        let inverted = invert_matrix(&self.gf, &a)
            .with_context(|| format!("Failed to invert matrix for survivors: {:?}", survivors))?;

        self.inverse_matrix_cache.insert(key, inverted.clone());
        Ok(inverted)
    }

    #[instrument(skip_all, fields(k = self.k, m = self.m, total_shards = shards_opt.len()))]
    pub fn reconstruct(&self, shards_opt: &mut [Option<Vec<u8>>]) -> Result<()> {
        assert_eq!(self.n, shards_opt.len());

        let shard_len = shards_opt
            .iter()
            .find_map(|s| s.as_ref().map(|v| v.len()))
            .ok_or_else(|| anyhow!("No shards available to determine length"))?;

        let present_indices: Vec<usize> =
            (0..self.n).filter(|&i| shards_opt[i].is_some()).collect();
        if present_indices.len() < self.k {
            return Err(anyhow!(
                "Not enough shards to reconstruct: have {}, need {}",
                present_indices.len(),
                self.k
            ));
        }

        let survivors = &present_indices[0..self.k];
        let a_inv = self.get_or_compute_inverse_matrix(survivors)?;

        let survivor_data: Vec<&[u8]> = survivors
            .iter()
            .map(|&idx| shards_opt[idx].as_ref().unwrap().as_slice())
            .collect();

        let missing_indices: Vec<usize> =
            (0..self.n).filter(|&i| shards_opt[i].is_none()).collect();
        if missing_indices.is_empty() {
            return Ok(());
        }

        let recovered_shards: Vec<(usize, Vec<u8>)> = missing_indices
            .par_iter()
            .map(|&missing_idx| {
                let _span = info_span!("reconstruct_shard", index = missing_idx).entered();
                let mut out_shard = vec![0u8; shard_len];

                let recovery_row = if missing_idx < self.k {
                    // If we're recovering a data shard, the recovery row is simply
                    // the corresponding row from the inverted matrix.
                    a_inv[missing_idx].clone()
                } else {
                    // If we're recovering a parity shard, we need to multiply its
                    // corresponding row from the original encoding matrix by the
                    // inverted matrix.
                    let encode_row = &self.encode_matrix[missing_idx - self.k];
                    mul_vec_matrix(&self.gf, encode_row, &a_inv)
                };

                for (j, sdata) in survivor_data.iter().enumerate() {
                    let coef = recovery_row[j];
                    if coef == 0 {
                        continue;
                    }

                    if coef == 1 {
                        for (out_byte, &in_byte) in out_shard.iter_mut().zip(sdata.iter()) {
                            *out_byte ^= in_byte;
                        }
                    } else {
                        let mult_table = self.gf.mul_table(coef);
                        for (out_byte, &in_byte) in out_shard.iter_mut().zip(sdata.iter()) {
                            *out_byte ^= mult_table[in_byte as usize];
                        }
                    }
                }
                (missing_idx, out_shard)
            })
            .collect();

        for (idx, shard_data) in recovered_shards {
            shards_opt[idx] = Some(shard_data);
        }

        Ok(())
    }
}
