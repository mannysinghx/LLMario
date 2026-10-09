//! Scalar reference implementations of the element-wise and attention primitives.
//!
//! These are deliberately plain loops. They are correct by inspection, serve as the oracle for
//! the SIMD kernels, and are fast enough for norms, RoPE and softmax (which are not where decode
//! time goes; the quantized matmuls are).

/// RoPE flavours the dense families use (Architecture §7.2 `RopeSpec`; extended in M3).
#[derive(serde::Serialize, serde::Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RopeKind {
    /// Adjacent pairs `(x[2i], x[2i+1])` rotate together (GPT-J style; llama.cpp `rope_type` 0).
    Normal,
    /// First and second halves pair up `(x[i], x[i + d/2])` (GPT-NeoX style; llama.cpp type 2).
    Neox,
}

/// `y = x / rms(x) * w`, ggml's `rms_norm` followed by the weight multiply.
pub fn rms_norm(x: &[f32], w: &[f32], eps: f32, y: &mut [f32]) {
    debug_assert_eq!(x.len(), w.len());
    debug_assert_eq!(x.len(), y.len());
    let mut ss = 0f32;
    for &v in x {
        ss += v * v;
    }
    let scale = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    for i in 0..x.len() {
        y[i] = x[i] * scale * w[i];
    }
}

/// Parameters of one RoPE application.
#[derive(Clone, Copy, Debug)]
pub struct RopeParams {
    pub kind: RopeKind,
    pub head_dim: usize,
    /// Number of dimensions rotated (≤ head_dim; "partial rotary").
    pub rot_dim: usize,
    pub theta: f32,
    /// Linear position scale (1.0 = none); YaRN and Llama-3 scaling arrive in M3.
    pub freq_scale: f32,
    /// Attention scale applied to the rotated values (YaRN `mscale`; 1.0 = none).
    pub attn_factor: f32,
}

/// Rotate `n_heads` heads of `head_dim` floats in place for absolute position `pos`.
pub fn rope(x: &mut [f32], n_heads: usize, pos: u32, p: &RopeParams) {
    debug_assert_eq!(x.len(), n_heads * p.head_dim);
    let half = p.rot_dim / 2;
    for h in 0..n_heads {
        let head = &mut x[h * p.head_dim..(h + 1) * p.head_dim];
        for i in 0..half {
            let freq = p.theta.powf(-(2.0 * i as f32) / p.rot_dim as f32);
            let angle = pos as f32 * p.freq_scale * freq;
            let (sin, cos) = angle.sin_cos();
            let (cos, sin) = (cos * p.attn_factor, sin * p.attn_factor);
            let (a, b) = match p.kind {
                RopeKind::Normal => (2 * i, 2 * i + 1),
                RopeKind::Neox => (i, i + half),
            };
            let x0 = head[a];
            let x1 = head[b];
            head[a] = x0 * cos - x1 * sin;
            head[b] = x0 * sin + x1 * cos;
        }
    }
}

/// In-place softmax over `x` (numerically stable).
pub fn softmax(x: &mut [f32]) {
    let max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// `out[i] = silu(gate[i]) * up[i]` (SwiGLU gating).
pub fn swiglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = silu(gate[i]) * up[i];
    }
}

/// `gate[i] = silu(gate[i]) * up[i]` in place.
pub fn swiglu_inplace(gate: &mut [f32], up: &[f32]) {
    for i in 0..gate.len() {
        gate[i] = silu(gate[i]) * up[i];
    }
}

/// tanh-approximated GELU (ggml `gelu`), used by GeGLU families.
#[inline]
pub fn gelu(x: f32) -> f32 {
    const C: f32 = 0.797_884_5; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

/// `out[i] = gelu(gate[i]) * up[i]`.
pub fn geglu(gate: &[f32], up: &[f32], out: &mut [f32]) {
    for i in 0..out.len() {
        out[i] = gelu(gate[i]) * up[i];
    }
}

/// `x += y`
pub fn add_inplace(x: &mut [f32], y: &[f32]) {
    for i in 0..x.len() {
        x[i] += y[i];
    }
}

/// Dot product of two f32 slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut s = 0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rms_norm_unit_vector() {
        let x = [3.0, 4.0];
        let w = [1.0, 1.0];
        let mut y = [0.0; 2];
        rms_norm(&x, &w, 0.0, &mut y);
        // rms = sqrt((9+16)/2) = 3.5355
        assert!((y[0] - 3.0 / 3.535_534).abs() < 1e-5);
    }

    #[test]
    fn rope_zero_position_is_identity() {
        let mut x: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let p = RopeParams {
            kind: RopeKind::Neox,
            head_dim: 8,
            rot_dim: 8,
            theta: 10000.0,
            freq_scale: 1.0,
            attn_factor: 1.0,
        };
        let orig = x.clone();
        rope(&mut x, 1, 0, &p);
        assert_eq!(x, orig);
    }

    #[test]
    fn rope_preserves_norm() {
        let mut x: Vec<f32> = (0..16).map(|i| (i as f32).sin()).collect();
        let n0 = dot(&x, &x);
        for kind in [RopeKind::Normal, RopeKind::Neox] {
            let p = RopeParams {
                kind,
                head_dim: 8,
                rot_dim: 8,
                theta: 10000.0,
                freq_scale: 1.0,
                attn_factor: 1.0,
            };
            let mut y = x.clone();
            rope(&mut y, 2, 37, &p);
            assert!((dot(&y, &y) - n0).abs() < 1e-4);
        }
        x[0] += 0.0;
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0, 2.0, 3.0, 1000.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(x[3] > 0.99);
    }

    #[test]
    fn activations() {
        assert!((silu(0.0)).abs() < 1e-7);
        assert!((gelu(0.0)).abs() < 1e-7);
        assert!((gelu(10.0) - 10.0).abs() < 1e-3);
        let mut o = [0.0; 2];
        swiglu(&[1.0, -1.0], &[2.0, 2.0], &mut o);
        assert!((o[0] - 2.0 * silu(1.0)).abs() < 1e-6);
    }
}
