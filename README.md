# Differentiable Floating-point Operations

The DFO crate aims at making your already existing code differentiable, leveraging automatic differentiation.

Currently, my goal is to implement the `Float` trait on `DFloat` for forward autodiff.

TODOs:

- [ ] Implement missing traits from `std::ops`
- [ ] Remove `Copy` trait when possible, especially for arrays
- [ ] Create utils functions to create constants, variables, etc.
- [ ] Create tests and benchmarks
- [ ] Implement other traits, until `Float` is implemented
- [ ] Reorganize and clean crate
- [ ] Create some GitHub actions
- [x] Publish crate on `https://crates.io>`
- [ ] Create examples for multiple inputs, multiple outputs, ndarrays, and more
- [ ] Go for backward autodiff

Similar crates:

- `<https://github.com/elrnv/autodiff>`
