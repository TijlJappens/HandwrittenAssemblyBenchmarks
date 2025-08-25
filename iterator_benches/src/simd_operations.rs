use std::arch::x86_64::__m128;
use std::arch::asm;

#[inline(always)]
pub fn sine_cosine_simd(mut angle: __m128) -> (__m128, __m128) {
    let mut cosine: __m128 = unsafe { std::arch::x86_64::_mm_set1_ps(1.0_f32) };
    unsafe {
        asm!(
            // xmm2 = angle^2
            "movaps {2}, {0}",
            "mulps {2}, {0}",
            // xmm3 = angle^3 = angle * angle^2
            "movaps {3}, {2}",
            "mulps {3}, {0}",

            // Sine: Horner's method with FMA
            // xmm4 = tmp = angle^2 * (1/120) + (-1/6)
            "movaps {4}, {9}",           // {9} = 1/120
            "vfmadd213ps {4}, {2}, {8}", // {4} = {4} * {2} + {8}, {8} = -1/6
            // xmm5 = angle^3 * tmp
            "movaps {5}, {3}",
            "mulps {5}, {4}",
            // xmm0 = angle + angle^3 * tmp
            "addps {0}, {5}",

            // Cosine: Horner's method with FMA
            // xmm6 = tmp = angle^2 * (1/24) + (-1/2)
            "movaps {6}, {11}",           // {11} = 1/24
            "vfmadd213ps {6}, {2}, {10}",  // {6} = {6} * {2} + {10}, {10} = -1/2
            // xmm7 = angle^2 * tmp
            "movaps {7}, {2}",
            "mulps {7}, {6}",
            // xmm1 = 1.0 + angle^2 * tmp
            "addps {1}, {7}",

            inout(xmm_reg) angle, // 0
            inout(xmm_reg) cosine, // 1

            out(xmm_reg) _, // 2 angle^2
            out(xmm_reg) _, // 3 angle^3
            out(xmm_reg) _, // 4 tmp for sine
            out(xmm_reg) _, // 5 angle^3 * tmp
            out(xmm_reg) _, // 6 tmp for cosine
            out(xmm_reg) _, // 7 angle^2 * tmp

            in(xmm_reg) std::arch::x86_64::_mm_set1_ps(-1.0_f32/6.0_f32),  // 8 (for sine)
            in(xmm_reg) std::arch::x86_64::_mm_set1_ps(1.0_f32/120.0_f32), // 9 (for sine)

            in(xmm_reg) std::arch::x86_64::_mm_set1_ps(-1.0_f32/2.0_f32),  // 10 (for cosine)
            in(xmm_reg) std::arch::x86_64::_mm_set1_ps(1.0_f32/24.0_f32),  // 11 (for cosine)

            options(nostack, nomem, pure)
        );
    }
    (angle, cosine)
}

#[inline(always)]
/// This function is without using horners method and vfmadd213ps
pub fn sine_cosine_simd_old(mut angle: __m128) -> (__m128, __m128) {
    // We will store sine in angle and the cosine in a second variable
    let mut cosine: __m128 = unsafe { std::arch::x86_64::_mm_set1_ps(1.0_f32) };
    unsafe {
        // angle = angle - angle^3 / 6.0 + angle^5 / 120.0
        // cosine = 1.0 - angle^2 / 2.0 + angle^4 / 24.0
        asm!(
            // xmm2 = angle^2
            "movaps {2}, {0}",
            "mulps {2}, {0}",
            // xmm3 = angle^3 = angle * angle^2
            "movaps {3}, {2}",
            "mulps {3}, {0}",
            // xmm4 = angle^4 = angle^2 * angle^2
            "movaps {4}, {2}",
            "mulps {4}, {2}",
            // xmm5 = angle^5
            "movaps {5}, {4}",
            "mulps {5}, {0}",

            // xmm2 = -angle^2 / 2.0
            "mulps {2}, {6}",
            // xmm3 = -angle^3 / 6.0
            "mulps {3},  {8}",
            // xmm4 = angle^4 / 24.0
            "mulps {4}, {7}",
            // xmm5 = angle^5 / 120.0
            "mulps {5}, {9}",

            // angle = angle - angle^3 / 6.0 + angle^5 / 120.0
            "addps {0}, {3}",
            "addps {0}, {5}",

            // cosine = 1.0 - angle^2 / 2.0 + angle^4 / 24.0
            "addps {1}, {4}",
            "addps {1}, {2}",

            // Output variables
            inout(xmm_reg) angle, // 0
            inout(xmm_reg) cosine, // 1

            // Dummy variables
            out(xmm_reg) _, // 2
            out(xmm_reg) _, // 3
            out(xmm_reg) _, // 4
            out(xmm_reg) _, // 5

            // constants
            in(xmm_reg) std::arch::x86_64::_mm_set1_ps(-1.0_f32/2.0_f32), // 6
            in(xmm_reg) std::arch::x86_64::_mm_set1_ps(1.0_f32/24.0_f32), // 7
            in(xmm_reg) std::arch::x86_64::_mm_set1_ps(-1.0_f32/6.0_f32), // 8
            in(xmm_reg) std::arch::x86_64::_mm_set1_ps(1.0_f32/120.0_f32), // 9
            options(nostack, nomem, pure)
        );
    }
    (angle, cosine)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{sine, cosine};

    #[test]
    fn test_sine_cosine_simd_matches_scalar() {
        let angles = [0.0_f32, 0.5_f32, 1.0_f32, 2.0_f32];
        let simd_angles = unsafe { std::arch::x86_64::_mm_set_ps(angles[3], angles[2], angles[1], angles[0]) };
        let (simd_sine, simd_cosine) = sine_cosine_simd(simd_angles);
        let mut simd_sine_arr = [0.0_f32; 4];
        let mut simd_cosine_arr = [0.0_f32; 4];
        unsafe {
            std::arch::x86_64::_mm_storeu_ps(simd_sine_arr.as_mut_ptr(), simd_sine);
            std::arch::x86_64::_mm_storeu_ps(simd_cosine_arr.as_mut_ptr(), simd_cosine);
        }
        for i in 0..4 {
            let scalar_sine = sine(angles[i]);
            let scalar_cosine = cosine(angles[i]);
            assert!((simd_sine_arr[i] - scalar_sine).abs() < 1e-5, "sine mismatch at {}: simd={}, scalar={}", i, simd_sine_arr[i], scalar_sine);
            assert!((simd_cosine_arr[i] - scalar_cosine).abs() < 1e-5, "cosine mismatch at {}: simd={}, scalar={}", i, simd_cosine_arr[i], scalar_cosine);
        }
    }
}