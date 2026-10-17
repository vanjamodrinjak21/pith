//! int8 dot-product kernels. The only `unsafe` in the crate lives here.
#![deny(unsafe_op_in_unsafe_fn)]

fn sum_scalar(a: &[i8], b: &[i8]) -> i32 {
    a.iter().zip(b).map(|(&x, &y)| x as i32 * y as i32).sum()
}

/// Portable reference kernel.
#[cfg(test)]
pub(crate) fn dot_i8_scalar(a: &[i8], b: &[i8]) -> i32 {
    assert_eq!(a.len(), b.len());
    sum_scalar(a, b)
}

/// int8 dot product using the best SIMD path for this CPU.
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    let mut out = 0;
    assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0;
    }
    dot_i8_rows(a, b, |_, s| out = s);
    out
}

/// Calls `f(row_index, dot(q, row))` for each `q.len()`-sized row of `rows`.
/// CPU dispatch and length checks happen once per call, not per row.
pub(crate) fn dot_i8_rows(q: &[i8], rows: &[i8], mut f: impl FnMut(usize, i32)) {
    assert!(!q.is_empty() && rows.len().is_multiple_of(q.len()));
    let it = rows.chunks_exact(q.len()).enumerate();
    #[cfg(target_arch = "aarch64")]
    for (i, row) in it {
        // SAFETY: NEON is mandatory on aarch64; chunks_exact gives row.len() == q.len().
        f(i, unsafe { dot_neon(q, row) });
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            for (i, row) in it {
                // SAFETY: AVX2 verified at runtime above; row.len() == q.len().
                f(i, unsafe { dot_avx2(q, row) });
            }
        } else {
            for (i, row) in it {
                f(i, sum_scalar(q, row));
            }
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    for (i, row) in it {
        f(i, sum_scalar(q, row));
    }
}

/// # Safety
/// Requires `a.len() == b.len()` (NEON is always present on aarch64).
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::aarch64::*;
    let n = a.len() / 16 * 16;
    let mut acc = vdupq_n_s32(0);
    let mut i = 0;
    while i < n {
        // SAFETY: i + 16 <= n <= a.len() == b.len(), so both 16-byte loads are in bounds.
        let (x, y) = unsafe { (vld1q_s8(a.as_ptr().add(i)), vld1q_s8(b.as_ptr().add(i))) };
        // i8*i8 fits in i16 (max 16384); pairwise-add into i32 lanes.
        acc = vpadalq_s16(acc, vmull_s8(vget_low_s8(x), vget_low_s8(y)));
        acc = vpadalq_s16(acc, vmull_high_s8(x, y));
        i += 16;
    }
    vaddvq_s32(acc) + sum_scalar(&a[n..], &b[n..])
}

/// # Safety
/// Requires AVX2 on the running CPU and `a.len() == b.len()`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_avx2(a: &[i8], b: &[i8]) -> i32 {
    use std::arch::x86_64::*;
    let n = a.len() / 16 * 16;
    let mut acc = _mm256_setzero_si256();
    let mut i = 0;
    while i < n {
        // SAFETY: i + 16 <= n <= a.len() == b.len(); loadu has no alignment requirement.
        let (x, y) = unsafe {
            (
                _mm_loadu_si128(a.as_ptr().add(i) as *const __m128i),
                _mm_loadu_si128(b.as_ptr().add(i) as *const __m128i),
            )
        };
        let (x, y) = (_mm256_cvtepi8_epi16(x), _mm256_cvtepi8_epi16(y));
        // madd: pairs of i16 products summed into i32 (max 32768, no overflow).
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(x, y));
        i += 16;
    }
    let s = _mm_add_epi32(
        _mm256_castsi256_si128(acc),
        _mm256_extracti128_si256(acc, 1),
    );
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
    let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
    _mm_cvtsi128_si32(s) + sum_scalar(&a[n..], &b[n..])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cases() -> Vec<(Vec<i8>, Vec<i8>)> {
        let mut s: u64 = 42;
        let mut gen = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 56) as u8 as i8
        };
        let mut out = vec![(vec![-128i8; 384], vec![-128i8; 384])];
        for len in [0usize, 1, 7, 15, 16, 31, 32, 33, 100, 384, 385] {
            for _ in 0..20 {
                out.push((
                    (0..len).map(|_| gen()).collect(),
                    (0..len).map(|_| gen()).collect(),
                ));
            }
        }
        out
    }

    #[test]
    fn simd_kernel_equals_scalar_kernel() {
        for (a, b) in cases() {
            let want = dot_i8_scalar(&a, &b);
            assert_eq!(dot_i8(&a, &b), want);
            #[cfg(target_arch = "aarch64")]
            // SAFETY: NEON is mandatory on aarch64; lengths are equal.
            assert_eq!(unsafe { dot_neon(&a, &b) }, want, "neon len {}", a.len());
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                // SAFETY: AVX2 verified at runtime; lengths are equal.
                assert_eq!(unsafe { dot_avx2(&a, &b) }, want, "avx2 len {}", a.len());
            }
        }
        // Batch path must match per-row scalar results exactly.
        let hundred: Vec<_> = cases().into_iter().filter(|c| c.0.len() == 100).collect();
        let q = hundred[0].0.clone();
        let rows: Vec<i8> = hundred.iter().flat_map(|c| c.1.clone()).collect();
        let mut got = vec![];
        dot_i8_rows(&q, &rows, |i, s| got.push((i, s)));
        let want: Vec<(usize, i32)> = rows
            .chunks_exact(100)
            .map(|r| dot_i8_scalar(&q, r))
            .enumerate()
            .collect();
        assert_eq!(got, want);
        #[cfg(target_arch = "x86_64")]
        if !std::arch::is_x86_feature_detected!("avx2") {
            println!("skipped: no avx2");
        } else {
            println!("ran: avx2 kernel");
        }
    }
}
