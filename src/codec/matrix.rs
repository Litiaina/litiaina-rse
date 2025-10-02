use crate::algorithm::gf256::Gf256;
use anyhow::{anyhow, Result};

pub type Matrix = Vec<Vec<u8>>;

pub fn mul_vec_matrix(gf: &Gf256, vec: &[u8], mat: &Matrix) -> Vec<u8> {
    let k = mat.len();
    assert_ne!(k, 0, "Matrix cannot be empty");
    let cols = mat[0].len();
    assert_eq!(vec.len(), k, "Vector length must match matrix rows");

    let mut result = vec![0u8; cols];
    for j in 0..cols {
        let mut sum = 0;
        for (i, &v_val) in vec.iter().enumerate() {
            sum ^= gf.mul(v_val, mat[i][j]);
        }
        result[j] = sum;
    }
    result
}

pub fn invert_matrix(gf: &Gf256, mat: &[Vec<u8>]) -> Result<Matrix> {
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

pub fn build_vandermonde(gf: &Gf256, k: usize, m: usize) -> Matrix {
    let mut matrix = vec![vec![0u8; k]; m];
    for r in 0..m {
        for c in 0..k {
            // Using (r + k) as x value to ensure it's not 0 or 1,
            // which can create degenerate matrices for some k,m values.
            matrix[r][c] = gf.exp[((r + k) as i32 * c as i32 % 255) as usize];
        }
    }
    matrix
}
