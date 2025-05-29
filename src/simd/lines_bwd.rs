// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

use std::ptr;

use crate::helpers::CoordType;

/// Starting from the `offset` in `haystack` with a current line index of
/// `line`, this seeks backwards to the `line_stop`-nth line and returns the
/// new offset and the line index at that point.
///
/// It returns an offset *past* the newline (= at the start of the next line).
pub fn lines_bwd(
    haystack: &[u8],
    offset: usize,
    line: CoordType,
    line_stop: CoordType,
) -> (usize, CoordType) {
    unsafe {
        let beg = haystack.as_ptr();
        let it = beg.add(offset.min(haystack.len()));
        let (it, line) = lines_bwd_raw(beg, it, line, line_stop);
        (it.offset_from_unsigned(beg), line)
    }
}

#[inline(never)]
unsafe fn lines_bwd_raw(
    beg: *const u8,
    end: *const u8,
    line: CoordType,
    line_stop: CoordType,
) -> (*const u8, CoordType) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    return unsafe { LINES_BWD_DISPATCH(beg, end, line, line_stop) };

    #[cfg(target_arch = "aarch64")]
    return unsafe { lines_bwd_neon(beg, end, line, line_stop) };

    #[allow(unreachable_code)]
    return unsafe { lines_bwd_fallback(beg, end, line, line_stop) };
}

#[inline]
unsafe fn lines_bwd_fallback(
    beg: *const u8,
    mut end: *const u8,
    mut line: CoordType,
    line_stop: CoordType,
) -> (*const u8, CoordType) {
    unsafe {
        if line > line_stop {
            while !ptr::eq(end, beg) {
                let n = end.sub(1);
                if *n == b'\n' {
                    line -= 1;
                    if line == line_stop {
                        break;
                    }
                }
                end = n;
            }
        }
        (end, line)
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
static mut LINES_BWD_DISPATCH: unsafe fn(
    beg: *const u8,
    end: *const u8,
    line: CoordType,
    line_stop: CoordType,
) -> (*const u8, CoordType) = lines_bwd_dispatch;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
unsafe fn lines_bwd_dispatch(
    beg: *const u8,
    end: *const u8,
    line: CoordType,
    line_stop: CoordType,
) -> (*const u8, CoordType) {
    let func = if is_x86_feature_detected!("avx2") { lines_bwd_avx2 } else { lines_bwd_fallback };
    unsafe { LINES_BWD_DISPATCH = func };
    unsafe { func(beg, end, line, line_stop) }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn lines_bwd_avx2(
    beg: *const u8,
    mut end: *const u8,
    mut line: CoordType,
    line_stop: CoordType,
) -> (*const u8, CoordType) {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        if end.offset_from_unsigned(beg) >= 32 {
            let n = _mm256_set1_epi8(b'\n' as i8);

            loop {
                let v = _mm256_loadu_si256(end.sub(32) as *const _);
                let c = _mm256_cmpeq_epi8(v, n);
                let m = _mm256_movemask_epi8(c).cast_unsigned();
                let line2 = line - m.count_ones() as CoordType;

                if line2 < line_stop {
                    break;
                }

                end = end.sub(32);
                line = line2;

                if end.offset_from_unsigned(beg) < 32 {
                    break;
                }
            }
        }

        lines_bwd_fallback(beg, end, line, line_stop)
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn lines_bwd_neon(
    beg: *const u8,
    mut end: *const u8,
    mut line: CoordType,
    line_stop: CoordType,
) -> (*const u8, CoordType) {
    unsafe {
        use std::arch::aarch64::*;

        if end.offset_from_unsigned(beg) >= 16 {
            let n1 = vdupq_n_u8(b'\n');

            loop {
                end = end.sub(16);

                let v = vld1q_u8(end as *const _);
            }
        }

        lines_bwd_fallback(beg, end)
    }
}
