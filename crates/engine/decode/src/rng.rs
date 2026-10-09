//! A small deterministic PRNG for the sampler: xoshiro256** seeded through splitmix64.
//!
//! The engine needs reproducible draws for a given `seed` (the API's `seed` parameter) without
//! pulling in an external RNG crate. xoshiro256** (Blackman & Vigna, public domain) is the
//! generator behind `rand`'s `SmallRng` on 64-bit targets; splitmix64 expands the 64-bit seed into
//! the 256-bit state so that every seed — including 0 — yields a non-zero state.

/// xoshiro256** generator state.
#[derive(Clone, Debug)]
pub struct Xoshiro256StarStar {
    s: [u64; 4],
}

/// One splitmix64 step: advances `state` and returns the next output.
pub(crate) fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Xoshiro256StarStar {
    /// Seeds the generator from a 64-bit seed. Equal seeds give equal streams.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Self { s }
    }

    /// Next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// A uniform `f64` in `[0, 1)` built from the top 53 bits of the next output.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix64_reference_vector() {
        // Published test values for splitmix64 seeded with 0.
        let mut s = 0u64;
        assert_eq!(splitmix64(&mut s), 0xE220_A839_7B1D_CDAF);
        assert_eq!(splitmix64(&mut s), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(splitmix64(&mut s), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn same_seed_same_stream_and_unit_range() {
        let mut a = Xoshiro256StarStar::seed_from_u64(42);
        let mut b = Xoshiro256StarStar::seed_from_u64(42);
        let mut c = Xoshiro256StarStar::seed_from_u64(43);
        let mut differs = false;
        for _ in 0..1000 {
            let x = a.next_u64();
            assert_eq!(x, b.next_u64());
            differs |= x != c.next_u64();
            let u = a.next_f64();
            assert!((0.0..1.0).contains(&u));
            b.next_f64();
            c.next_f64();
        }
        assert!(differs, "different seeds should produce different streams");
    }
}
