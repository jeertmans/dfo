use criterion::criterion_main;

mod compare_autodiff;
mod measure_overhead;

criterion_main! {
    compare_autodiff::benches,
    measure_overhead::benches,
}
