use anyhow::{Result, anyhow};

#[derive(Debug, Clone)]
pub struct Gf256 {
    pub exp: Vec<u8>,
    pub log: Vec<i16>,
}

impl Gf256 {
    pub fn new() -> Self {
        let mut exp = vec![0u8; 512];
        let mut log = vec![-1i16; 256];
        let mut x: u16 = 1;

        for i in 0..255 {
            exp[i] = x as u8;
            log[x as usize] = i as i16;
            x <<= 1;
            if x & 0x100 != 0 {
                x ^= 0x11d;
            }
        }
        for i in 255..512 {
            exp[i] = exp[i - 255];
        }
        Gf256 { exp, log }
    }

    pub fn mul_table(&self, factor: u8) -> [u8; 256] {
        let mut table = [0u8; 256];
        if factor == 0 {
            return table;
        }
        let log_factor = self.log[factor as usize] as i32;
        for i in 0..=255 {
            if i > 0 {
                let log_i = self.log[i as usize] as i32;
                table[i as usize] = self.exp[(log_i + log_factor) as usize];
            }
        }
        table
    }

    #[inline]
    pub fn mul(&self, a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            0
        } else {
            let la = self.log[a as usize] as i32;
            let lb = self.log[b as usize] as i32;
            self.exp[(la + lb) as usize]
        }
    }

    #[inline]
    pub fn inv(&self, a: u8) -> Result<u8> {
        if a == 0 {
            return Err(anyhow!("inverse of zero is undefined"));
        }
        let la = self.log[a as usize] as i32;
        Ok(self.exp[(255 - la) as usize])
    }
}

impl Default for Gf256 {
    fn default() -> Self {
        Self::new()
    }
}
