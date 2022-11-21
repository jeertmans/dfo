/// Trait for any number-like structure that implements
/// automatic foward differentiation.
/// 
/// A variable should start with unit derivative, and
/// a constant should have a zero derivative.
///
/// # Examples
///
/// ```
/// # use dfo::forward::primitive::*;
/// let x = DFloat32::var(14.0);
///
/// assert_eq!(*x.value(), 14.0);
/// assert_eq!(*x.deriv(),  1.0);
///
/// let c = DFloat32::cst(3.14);
///
/// assert_eq!(*c.value(), 3.14);
/// assert_eq!(*c.deriv(),  0.0);
///
/// let y = c * x * x; // y = c * x^2
///
/// assert_eq!(*y.deriv(), 2.0 * 3.14 * 14.0);
/// ```
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
    /// Creates a new number from tuple.
    fn from_tuple(x: Self::Inner, dx: Self::Inner) -> Self;
    /// Destructures self into a tuple.
    fn into_tuple(self) -> (Self::Inner, Self::Inner);
}
