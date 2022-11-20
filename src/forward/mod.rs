//! Forward automatic differentiation.
pub mod primitive;
mod traits;

pub use primitive::{DFloat32, DFloat64};
pub use traits::Differentiable;
