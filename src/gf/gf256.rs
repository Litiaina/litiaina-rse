use anyhow::{Result, anyhow};

#[derive(Debug, Clone)]
pub struct Gf256 {
    exp: Vec<u8>,
    log: Vec<i16>,
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
