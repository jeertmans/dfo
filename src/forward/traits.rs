/// Trait for any number-like structure that implements
/// automatic foward differentiation.
pub trait Differentiable {
    /// Inner type, that can be used to construct
    /// a new differentiable variable of constant.
    type Inner;
    /// Creates a new variable, whose derivative will
    /// propagate.
    fn var(x: Self::Inner) -> Self;
    /// Creates a new constant, whose derivative will
    /// not propagate, since a constant has a zero derivative.
    fn cst(x: Self::Inner) -> Self;
    /// Returns the inner value of this number.
    ///
    /// Usually, this is the same value as if no derivative were
    /// ever computed.
    fn value(&self) -> &Self::Inner;
    /// Returns the derivative of this number.
    fn deriv(&self) -> &Self::Inner;
}
