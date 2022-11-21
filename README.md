# Differentiable Floating-point Operations

&#9888; This crate is very much in progress and should not be used as if!

If you want to contribute, or ask a question, please contact me through [GitHub issues](https://github.com/jeertmans/dfo/issues).


[![Crates.io](https://img.shields.io/crates/v/dfo)](https://crates.io/crates/dfo)
[![docs.rs](https://img.shields.io/docsrs/dfo)](https://docs.rs/dfo)

The DFO crate aims at making your already existing code differentiable, leveraging automatic differentiation.

## Crate Structure

As for automatic differentiation, this crate is made of two separate parts:

- the [`forward`] module;
- and the [`backward`] module.

As their name implies, the former implements *forward*-mode automatic differentiation, while the latter implements *backward*-mode.

> **NOTE:** if you are new to automatic differentation, I recommend you to read [this excellent article from Max Slater's blog](https://thenumb.at/Autodiff/).

Next, each of those modules is split in (at least) two submodules: *primitive* and *generic*.

The primitive module implements a given differentiation mode on primitive types, e.g., [`f32`] with [`forward::DFloat32`].

### Example - Forward auto. diff. on primitive types

```rust
use dfo::forward::primitive::*;

let f = |x| { x * x + 1.0 - x };
let df = |x| { 2.0 * x - 1.0 };
let x = DFloat32::var(4.0);

assert_eq!(f(x).deriv(), df(x).value());
```

The generic module implements a given differentiation mode on generic types. For the latter, we recommend using the [`num_traits::Float`](https://docs.rs/num-traits/latest/num_traits/float/trait.Float.html#) trait, for example. Primitive types are usually better optimized (no useless copy, faster compilation time), and recommended unless you want to work with external crates, as [`ndarray`](https://docs.rs/ndarray/latest/ndarray/).

## Crate Feature Flags

The following crate feature flags are available. They are configured in your `Cargo.toml`.

* `num-traits`: supports for (most) traits defined in [`num-traits`](https://docs.rs/num-traits/latest/num_traits/)
* `std`: Rust standard library-using functionality (enabled by default)
* `serde`: serialization support for [`serde`](https://serde.rs/) 1.x

## Crate Goals

DFO has two main objectives. It has to be:

1. **fast**, meaning reducing computations and copies when possible, intensive use of code inlining, and others;
2. and **easy-to-use**, leveraging generatic types, and requiring very few modifications to compute derivation.

## Crate Status

**\[TL;DR\]** Is this create stable? It it production-ready?

> No, and maybe not until a long time.

**Currently**, the following modules are implemented:

- [ ] forward
  * [x] primitive
  * [ ] generic
- [ ] backward
  * [ ] primitive
  * [ ] generic


Here is a small **TO-DO** list of things  that I would like to work on:

- [ ] Improve docs
- [ ] Benchmark against autodiffs
- [ ] Create features for `primitive`, `forward` and `backward`
- [ ] Add more needed impl. for Differentiable trait: to and from tuple
- [ ] Create tests and benchmarks
- [ ] Implement other traits, until `Float` is implemented
- [ ] Reorganize and clean crate
- [ ] Create examples for multiple inputs, multiple outputs, ndarrays, and more
- [ ] Go for backward autodiff

## Similar crates:

- [`autodiff`](https://github.com/elrnv/autodiff)
