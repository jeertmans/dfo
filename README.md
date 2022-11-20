# Differentiable Floating-point Operations

The DFO crate aims at making your already existing code differentiable, leveraging automatic differentiation.

Currently, my goal is to implement the `Float` trait on `DFloat` for forward autodiff.

TODOs:

- [ ] Remove `Copy` trait when possible, especially for arrays
- [ ] Make examples in docstring
- [ ] Improve docs
- [ ] Clean README
- [ ] Benchmark against autodiffs
- [ ] Create features for `primitive`, `forward` and `backward`
- [ ] Add more needed impl. for Differentiable trait: to and from tuple
- [ ] Create tests and benchmarks
- [ ] Implement other traits, until `Float` is implemented
- [ ] Reorganize and clean crate
- [ ] Create some GitHub actions
- [ ] Create examples for multiple inputs, multiple outputs, ndarrays, and more
- [ ] Go for backward autodiff

Similar crates:

- `<https://github.com/elrnv/autodiff>`
