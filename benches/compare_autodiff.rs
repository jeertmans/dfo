use autodiff::{Float as ADFloat, F, FT};
use criterion::{criterion_group, BenchmarkId, Criterion};
use dfo::forward::primitive::*;
use ndarray::{Array1};


fn autodiff_basic(x: f64) -> f64 {
    let x = F::var(x);
    let y: FT<_> = x.exp() * x.sin() + 7.0 * x * x + F::cst(1.0);
    y.deriv()
}

fn dfo_basic(x: f64) -> f64 {
    let x = DFloat64::var(x);
    let y = x.exp() * x.sin() + 7.0 * x * x + 1.0;
    *y.deriv()
}

fn autodiff_square(x: f64) -> f64 {
    let x: F<f64, f64> = F::var(x);
    (x * x).deriv()
}

fn dfo_square(x: f64) -> f64 {
    let x = DFloat64::var(x);
    *(x * x).deriv()
}

fn autodiff_loop(n: usize) -> f64 {
    let mut x: F<f64, f64> = F::var(0.9f64);

    for _i in 0..n {
        x = x * 0.9
    }
    x.deriv()
}

fn dfo_loop(n: usize) -> f64 {
    let mut x = DFloat64::var(0.9f64);

    for _i in 0..n {
        x = x * x
    }
    *x.deriv()
}

fn autodiff_loop_cst(n: usize) -> f64 {
    let mut x: F<f64, f64> = F::var(2f64);

    for _i in 0..n {
        x = x * 0.9
    }
    x.deriv()
}

fn dfo_loop_cst(n: usize) -> f64 {
    let mut x = DFloat64::var(2f64);

    for _i in 0..n {
        x = x * 0.9
    }
    *x.deriv()
}

fn autodiff_medium(x: f64) -> f64 {
    let x = F::var(x);
    let y: F<f64, f64> = x.exp().ln().cos().powi(3) * x.tanh() * 7.0 + 4.0 * x.powf(x).abs();
    y.deriv()
}

fn dfo_medium(x: f64) -> f64 {
    let x = DFloat64::var(x);
    let y = x.exp().ln().cos().powi(3) * x.tanh() * 7.0 + 4.0 * x.powf(x).abs();
    *y.deriv()
}

fn autodiff_array(n: usize) -> f64 {
    let x: F<f64, f64> = F::var(3.14f64);

    let ones: Array1<F<f64, f64>> = Array1::ones(n);

    let x: Array1<F<f64, f64>> = ones.map(|o| *o * x);

    let y: Array1<F<f64, f64>> = x.map(|x| *x * x.exp() + 3.0); 
    y.mean().unwrap().deriv()
}

fn dfo_array(n: usize) -> f64 {
    let x = DFloat64::var(3.14f64);

    let ones: Array1<DFloat64> = Array1::ones(n);

    let x: Array1<DFloat64> = ones.map(|o| *o * x);

    let y: Array1<DFloat64> = x.map(|x| *x * x.exp() + 3.0); 
    *(y.mean().unwrap().deriv())
}


fn autodiff_array_product(n: usize) -> f64 {
    let x: F<f64, f64> = F::var(3.14f64);

    let ones: Array1<F<f64, f64>> = Array1::ones(n);

    let x: Array1<F<f64, f64>> = ones.map(|o| *o * x);

    let y: Array1<F<f64, f64>> = x.map(|x| x.cos()); 
    y.product().deriv()
}

fn dfo_array_product(n: usize) -> f64 {
    let x = DFloat64::var(3.14f64);

    let ones: Array1<DFloat64> = Array1::ones(n);

    let x: Array1<DFloat64> = ones.map(|o| *o * x);

    let y: Array1<DFloat64> = x.map(|x| x.cos()); 
    *(y.product().deriv())
}

macro_rules! bench_group (
    ($name:ident, $adfun:ident, $dfofun:ident $(, $inputs:literal)*) => (
        concat_idents::concat_idents!(bench_name = bench_, $name, {
            fn bench_name(c: &mut Criterion) {
                let mut group = c.benchmark_group(stringify!($name));
                for x in [$($inputs ,)*].iter() {
                    let left = $adfun(*x);
                    let right = $dfofun(*x);
                    assert!((left - right).abs() <= 1e-6,  "autodiff and dfo derivatives are not close enough: {:?} (ad) and {:?} (dfo)", left, right);
                    group.bench_with_input(BenchmarkId::new("autodiff", x), x,
                        |b, x| b.iter(|| $adfun(*x)));
                    group.bench_with_input(BenchmarkId::new("dfo", x), x,
                        |b, x| b.iter(|| $dfofun(*x)));
                }
                group.finish();
            }
        });
    );
);

bench_group!(basics, autodiff_basic, dfo_basic, 3f64);

bench_group!(squares, autodiff_square, dfo_square, 300f64);

bench_group!(loops, autodiff_loop, dfo_loop, 10000);

bench_group!(loops_cst, autodiff_loop_cst, dfo_loop_cst, 10000);

bench_group!(mediums, autodiff_medium, dfo_medium, 3f64);

bench_group!(arrays, autodiff_array, dfo_array, 10000);

bench_group!(arrays_product, autodiff_array_product, dfo_array_product, 10000);

criterion_group!(
    benches,
    bench_basics,
    bench_squares,
    bench_loops,
    bench_loops_cst,
    bench_mediums,
    bench_arrays,
    bench_arrays_product,
);
