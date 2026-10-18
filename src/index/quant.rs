//! Symmetric per-index int8 quantization.

/// Scale shared by the whole index: max |x| maps to 127.
pub fn scale_for(values: &[f32]) -> f32 {
    let m = values.iter().fold(0f32, |m, x| m.max(x.abs()));
    if m > 0.0 {
        m / 127.0
    } else {
        1.0
    }
}

pub fn quantize_one(x: f32, scale: f32) -> i8 {
    (x / scale).round().clamp(-127.0, 127.0) as i8
}

pub fn quantize(values: &[f32], scale: f32) -> Vec<i8> {
    values.iter().map(|&x| quantize_one(x, scale)).collect()
}

#[cfg(test)]
mod tests {
    #[test]
    fn scale_for_all_zero_is_finite_nonzero() {
        let s = super::scale_for(&[0.0; 8]);
        assert!(s.is_finite() && s > 0.0);
        assert_eq!(super::quantize(&[0.0; 8], s), vec![0; 8]);
        assert!(super::scale_for(&[]).is_finite());
    }
}
