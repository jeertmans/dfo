use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use dfo::forward::primitive::*;
use num_traits::Float;

fn value_basic(x: f64) -> f64 {
    x.exp() * x.sin() + 7.0 * x * x + 1.0
}

fn deriv_basic(x: f64) -> f64 {
    x.exp() * (x.sin() + x.cos()) + 14.0 * x
}

fn dfo_basic(x: f64) -> DFloat64 {
    let x = DFloat64::var(x);
    let y = x.exp() * x.sin() + 7.0 * x * x + 1.0;
    y
}

fn value_square(x: f64) -> f64 {
    x * x
}

fn deriv_square(x: f64) -> f64 {
    2.0 * x
}

fn dfo_square(x: f64) -> DFloat64 {
    let x = DFloat64::var(x);
    x * x
}

fn value_loop(n: usize) -> f64 {
    let mut x = 0.9f64;

    for _i in 0..n {
        x = x * x
    }
    x
}

fn deriv_loop(n: usize) -> f64 {
    let mut x = 0.9f64;

    for _i in 0..(n - 1) {
        x = x * x
    }
    (n as f64) * x
}

fn dfo_loop(n: usize) -> DFloat64 {
    let mut x = DFloat64::var(0.9f64);

    for _i in 0..n {
        x = x * x
    }
    x
}

fn value_loop_cst(n: usize) -> f64 {
    let mut x = 2f64;

    for _i in 0..n {
        x = x * 0.9
    }
    x
}

fn deriv_loop_cst(n: usize) -> f64 {
    let mut x = 1f64;

    for _i in 0..n {
        x = x * 0.9
    }
    x
}

fn dfo_loop_cst(n: usize) -> DFloat64 {
    let mut x = DFloat64::var(2f64);

    for _i in 0..n {
        x = x * 0.9
    }
    x
}

macro_rules! bench_group (
    ($name:ident, $vfun:ident, $dfun:ident, $dfofun:ident $(, $inputs:literal)*) => (
        concat_idents::concat_idents!(bench_name = bench_, $name, {
            fn bench_name(c: &mut Criterion) {
                let mut group = c.benchmark_group(format!("measure_overhead_{}", stringify!($name)));
                for x in [$($inputs ,)*].iter() {
                    let f = $vfun(*x);
                    let df = $dfun(*x);
                    let y = $dfofun(*x);
                    debug_assert!((f - *y.value()).abs() <= 1e-6,  "expected and dfo values are not close enough: {:?} (expected) and {:?} (dfo)", f, *y.value());
                    debug_assert!((df - *y.deriv()).abs() <= 1e-6,  "expected and dfo derivatives are not close enough: {:?} (expected) and {:?} (dfo)", df, *y.deriv());
                    group.bench_with_input(BenchmarkId::new("value", x), x,
                        |b, x| b.iter(|| $vfun(black_box(*x))));
                    group.bench_with_input(BenchmarkId::new("deriv", x), x,
                        |b, x| b.iter(|| $dfun(black_box(*x))));
                    group.bench_with_input(BenchmarkId::new("dfo", x), x,
                        |b, x| b.iter(|| $dfofun(black_box(*x))));
                }
                group.finish();
            }
        });
    );
);

bench_group!(basics, value_basic, deriv_basic, dfo_basic, 3f64);

bench_group!(squares, value_square, deriv_square, dfo_square, 300f64);

bench_group!(loops, value_loop, deriv_loop, dfo_loop, 10000);

bench_group!(
    loops_cst,
    value_loop_cst,
    deriv_loop_cst,
    dfo_loop_cst,
    10000
);

criterion_group!(
    benches,
    bench_basics,
    bench_squares,
    bench_loops,
    bench_loops_cst,
);

criterion_main!(benches);
