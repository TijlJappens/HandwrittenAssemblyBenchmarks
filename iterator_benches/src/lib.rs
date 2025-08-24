use std::arch::asm;

mod simd_operations;

#[inline(always)]
pub fn sine(angle: f32) -> f32 {
    let angle_squared = angle * angle;
    let angle_cubed = angle * angle_squared;
    let angle_fifth = angle_cubed * angle_squared;
    angle - angle_cubed * (1.0 / 6.0) + angle_fifth * (1.0 / 120.0)
}

#[inline(always)]
pub fn cosine(angle: f32) -> f32 {
    let angle_squared = angle * angle;
    let angle_fourth = angle_squared * angle_squared;
    1.0 - angle_squared * (1.0 / 2.0) + angle_fourth * (1.0 / 24.0)
}



pub fn my_function(slice: &[f32]) -> f32 {
    slice
        .iter()
        .map(|x| (sine(*x), cosine(*x)))
        .map(|(s, c)| s*s + c*c)
        .sum()
}

pub fn my_function_with_assembly(slice: &[f32]) -> f32 {
    let (chunks, remainder) = slice
        .as_chunks::<4>();

    let original_simd_value = unsafe { std::arch::x86_64::_mm_set_ps(0.0_f32, 0.0_f32, 0.0_f32, 0.0_f32) };

    let chunk_sum = chunks
        .iter()
        .map(|chunk| unsafe { std::arch::x86_64::_mm_set_ps(chunk[3], chunk[2], chunk[1], chunk[0]) })
        .fold(original_simd_value, |mut simd_value_so_far, chunk| {
            let (sine_simd, cosine_simd) = simd_operations::sine_cosine_simd(chunk);
            unsafe{
                asm!(
                    "mulps {1}, {1}",
                    "mulps {2}, {2}",
                    "addps {0}, {1}",
                    "addps {0}, {2}",
                    // Output variables
                    inout(xmm_reg) simd_value_so_far,
                    // Input variables
                    in(xmm_reg) sine_simd,
                    in(xmm_reg) cosine_simd,
                    options(nostack)
                );
            }
            simd_value_so_far
        });
    let mut arr: [f32; 4] = [0.0; 4];
    unsafe {
        std::arch::x86_64::_mm_store_ps(arr.as_mut_ptr(), chunk_sum);
    }
    let sum_from_chunk_sum: f32 = arr
        .into_iter()
        .sum();

    let remainder_sum = remainder
        .iter()
        .map(|x| (sine(*x), cosine(*x)))
        .map(|(s, c)| s*s + c*c)
        .sum::<f32>();
    sum_from_chunk_sum + remainder_sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_my_function_with_assembly_matches_scalar() {
        let data: Vec<f32> = (0..100).map(|x| x as f32 * 0.001).collect();
        let scalar = my_function(&data);
        let simd = my_function_with_assembly(&data);
        assert!((scalar - simd).abs() < 1e-4, "SIMD and scalar results differ: scalar={}, simd={}", scalar, simd);
    }
}