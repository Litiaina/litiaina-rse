use crate::gf::gf256::Gf256;
use anyhow::{Result, anyhow};

pub fn invert_matrix(gf: &Gf256, mat: &[Vec<u8>]) -> Result<Vec<Vec<u8>>> {
    let n = mat.len();
    if n == 0 || mat.iter().any(|r| r.len() != n) {
        return Err(anyhow!("Matrix must be square"));
    }

    let mut aug = (0..n)
        .map(|r| {
            let mut row = vec![0u8; 2 * n];
            row[..n].copy_from_slice(&mat[r]);
            row[n + r] = 1;
            row
        })
        .collect::<Vec<_>>();

    for col in 0..n {
        let pivot_row = (col..n)
            .find(|&r| aug[r][col] != 0)
            .ok_or_else(|| anyhow!("Matrix is singular and cannot be inverted"))?;
        aug.swap(col, pivot_row);

        let inv_pivot = gf.inv(aug[col][col])?;
        for j in col..(2 * n) {
            aug[col][j] = gf.mul(inv_pivot, aug[col][j]);
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            if factor == 0 {
                continue;
            }
            for j in col..(2 * n) {
                let prod = gf.mul(factor, aug[col][j]);
                aug[row][j] ^= prod;
            }
        }
    }

    let inv = aug.into_iter().map(|row| row[n..].to_vec()).collect();
    Ok(inv)
}

pub fn build_vandermonde(k: usize, m: usize) -> Vec<Vec<u8>> {
    let mut matrix = vec![vec![0u8; k]; m];
    let gf = Gf256::new();
    for r in 0..m {
        for c in 0..k {
            let mut val = 1;
            let x = (r + 1) as u8;
            for _ in 0..c {
                val = gf.mul(val, x);
            }
            matrix[r][c] = val;
        }
    }
    matrix
}
