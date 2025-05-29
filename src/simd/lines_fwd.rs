// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

use std::ptr;

use crate::helpers::CoordType;

/// Starting from the `offset` in `haystack` with a current line index of
/// `line`, this seeks to the `line_stop`-nth line and returns the
/// new offset and the line index at that point.
///
/// It returns an offset *past* the newline.
pub fn lines_fwd(
    haystack: &[u8],
    offset: usize,
    line: CoordType,
    line_stop: CoordType,
) -> (usize, CoordType) {
    unsafe {
        let beg = haystack.as_ptr();
        let end = beg.add(haystack.len());
        let it = beg.add(offset.min(haystack.len()));
        let (it, line) = lines_fwd_raw(it, end, line, line_stop);
        (it.offset_from_unsigned(beg), line)
    }
}

#[inline(never)]
unsafe fn lines_fwd_raw(
    beg: *const u8,
    end: *const u8,
    line: CoordType,
    line_stop: CoordType,
) -> (*const u8, CoordType) {
    #[cfg(target_arch = "x86_64")]
    return unsafe { LINES_FWD_DISPATCH(beg, end, line, line_stop) };

    #[cfg(target_arch = "aarch64")]
    return unsafe { lines_fwd_neon(beg, end, line, line_stop) };

    #[allow(unreachable_code)]
    return unsafe { lines_fwd_fallback(beg, end, line, line_stop) };
}

#[inline]
unsafe fn lines_fwd_fallback(
    mut beg: *const u8,
    end: *const u8,
    mut line: CoordType,
    line_stop: CoordType,
) -> (*const u8, CoordType) {
    unsafe {
        if line < line_stop {
            while !ptr::eq(beg, end) {
                let c = *beg;
                beg = beg.add(1);
                if c == b'\n' {
                    line += 1;
                    if line == line_stop {
                        break;
                    }
                }
            }
        }
        (beg, line)
    }
}

#[cfg(target_arch = "x86_64")]
static mut LINES_FWD_DISPATCH: unsafe fn(
    beg: *const u8,
    end: *const u8,
    line: CoordType,
    line_stop: CoordType,
) -> (*const u8, CoordType) = lines_fwd_dispatch;

#[cfg(target_arch = "x86_64")]
unsafe fn lines_fwd_dispatch(
    beg: *const u8,
    end: *const u8,
    line: CoordType,
    line_stop: CoordType,
) -> (*const u8, CoordType) {
    let func = if is_x86_feature_detected!("avx2") { lines_fwd_avx2 } else { lines_fwd_fallback };
    unsafe { LINES_FWD_DISPATCH = func };
    unsafe { func(beg, end, line, line_stop) }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn lines_fwd_avx2(
    mut beg: *const u8,
    end: *const u8,
    mut line: CoordType,
    line_stop: CoordType,
) -> (*const u8, CoordType) {
    unsafe {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let n = _mm256_set1_epi8(b'\n' as i8);
        let mut remaining = end.offset_from_unsigned(beg);

        if line < line_stop && remaining >= 32 {
            while remaining >= 32 {
                let v = _mm256_loadu_si256(beg as *const _);
                let c = _mm256_cmpeq_epi8(v, n);
                let m = _mm256_movemask_epi8(c).cast_unsigned();
                let line2 = line + m.count_ones() as CoordType;

                if line2 >= line_stop {
                    break;
                }

                beg = beg.add(32);
                remaining -= 32;
                line = line2;
            }
        }

        lines_fwd_fallback(beg, end, line, line_stop)
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn lines_fwd_neon(
    mut beg: *const u8,
    end: *const u8,
    count: CoordType,
) -> (*const u8, CoordType) {
    unsafe {
        use std::arch::aarch64::*;

        if end.offset_from_unsigned(beg) >= 16 {
            let n1 = vdupq_n_u8(b'\n');

            loop {
                let v = vld1q_u8(beg as *const _);
                let a = vceqq_u8(v, n1);
            }
        }

        lines_fwd_fallback(beg, end)
    }
}
