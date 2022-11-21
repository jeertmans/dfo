//! Forward automatic differentiation.
pub mod primitive;
//pub mod generic;
mod traits;

pub use primitive::{DFloat32, DFloat64};
pub use traits::Differentiable;
