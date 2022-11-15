/// Forward Mode Automatic Differentiation
///
///
use num_traits::{One, Zero};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DFloat<T> {
    x: T,
    dx: T,
}

impl<T: Zero> From<T> for DFloat<T> {
    fn from(x: T) -> Self {
        Self { x, dx: T::zero() }
    }
}

impl<T> DFloat<T> {
    pub fn value(&self) -> &T {
        &self.x
    }

    pub fn diff(&self) -> &T {
        &self.dx
    }
}

impl<T: One> DFloat<T> {
    pub fn var(x: T) -> Self {
        Self { x, dx: T::one() }
    }
}

pub type DFloat32 = DFloat<f32>;
pub type DFloat64 = DFloat<f64>;

// Implementing std::ops traits

macro_rules! impl_binop (
    ($trait:ident, $method:ident, |$xa:tt : $_xa:ty, $dxa:tt : $_dxa:ty, $xb:tt : $_xb:ty, $dxb:tt : $_dxb:ty| $body:block ) => (

impl<'a, A, B> $trait<&'a DFloat<B>> for &'a DFloat<A>
where
    A: $trait<B, Output = A>,
    &'a A: $trait<&'a B, Output = A>,
{
    type Output = DFloat<A>;
    fn $method(self, other: &'a DFloat<B>) -> Self::Output {
        let ($xa, $dxa) = (&self.x, &self.dx);
        let ($xb, $dxb) = (&other.x, &other.dx);
        $body
    }
}

impl<'a, A, B> $trait<DFloat<B>> for &'a DFloat<A>
where
    A: $trait<B, Output = B>,
    &'a A: $trait<B, Output = B>,
{
    type Output = DFloat<B>;
    fn $method(self, other: DFloat<B>) -> Self::Output {
        let other: DFloat<B> = other.into();
        let ($xa, $dxa) = (&self.x, &self.dx);
        let ($xb, $dxb) = (other.x, other.dx);
        $body
    }
}

impl<'a, A, B> $trait<&'a DFloat<B>> for DFloat<A>
where
    A: $trait<&'a B, Output = A>,
{
    type Output = DFloat<A>;
    fn $method(self, other: &'a DFloat<B>) -> Self::Output {
        let ($xa, $dxa) = (self.x, self.dx);
        let ($xb, $dxb) = (&other.x, &other.dx);
        $body
    }
}

impl<A, B> $trait<DFloat<B>> for DFloat<A>
where
    A: $trait<B, Output = A>,
{
    type Output = DFloat<A>;
    fn $method(self, other: DFloat<B>) -> Self::Output {
        let ($xa, $dxa) = (self.x, self.dx);
        let ($xb, $dxb) = (other.x, other.dx);
        $body
    }
}
);

    );

impl_binop!(Add, add, |xa: A, dxa: A, xb: B, dxb: B| {
    DFloat {
        x: xa + xb,
        dx: dxa + dxb,
    }
});
//impl_binop!(Div, div, |xa: A, dxa: A, xb: B, dxb: B| { DFloat { x: xa / xb, dx: dxa + dxb }});
//impl_binop!(Mul, mul, |xa: A, dxa: A, xb: B, dxb: B| { DFloat { x: xa * xb, dx: dxa + dxb }});
//impl_binop!(Sub, sub, |xa: A, dxa: A, xb: B, dxb: B| { DFloat { x: xa - xb, dx: dxa - dxb }});

#[cfg(test)]
mod tests {
    use super::*;
    use super::{DFloat32 as DF32, DFloat64 as DF64};
    use ndarray::{arr1, Array1};

    #[test]
    fn test_add() {
        let d1 = DFloat { x: 1.0, dx: 2.0 };
        let d2 = DFloat { x: 2.0, dx: 4.0 };

        assert_eq!(d1.clone() + d2.clone(), DFloat { x: 3.0, dx: 6.0});
        assert_eq!(d1.clone() + DF32::from(2.0), DFloat { x: 3.0, dx: 2.0});
        assert_eq!(&d1 + &d2, DFloat { x: 3.0, dx: 6.0});
        assert_eq!(d1.clone() + &d2, DFloat { x: 3.0, dx: 6.0});
        assert_eq!(&d1 + d2.clone(), DFloat { x: 3.0, dx: 6.0});

        let a1 = arr1(&vec![1, 2, 3, 4]);
        let a2 = arr1(&vec![5, 6, 7, 8]);

        let x = DFloat {
            x: a1.clone(),
            dx: a2.clone(),
        };

        let y = DFloat { x: &a2, dx: &a2 };

        let _: Array1<i32> = &a1 + a1.clone();

        let _: DFloat<Array1<i32>> = x.clone() + y.clone();

        //assert_eq!(&x + &y, &y + &x);
    }
}
