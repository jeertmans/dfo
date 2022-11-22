use criterion::criterion_main;

mod compare_autodiff;

criterion_main! {
    compare_autodiff::benches,
}
