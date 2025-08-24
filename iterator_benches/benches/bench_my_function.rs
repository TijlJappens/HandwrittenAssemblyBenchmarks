#![feature(test)]
extern crate test;
use iterator_benches::{my_function, my_function_with_assembly};
use test::Bencher;

#[bench]
fn bench_my_function_normal(b: &mut Bencher) {
    let large_vec: Vec<f32> = (1..10_000_001).map(|x| 1.0 / x as f32).collect();
    b.iter(|| {
        test::black_box(my_function(&large_vec));
    });
}

#[bench]
fn bench_my_function_with_assembly(b: &mut Bencher) {
    let large_vec: Vec<f32> = (1..10_000_001).map(|x| 1.0 / x as f32).collect();
    b.iter(|| {
        test::black_box(my_function_with_assembly(&large_vec));
    });
}
