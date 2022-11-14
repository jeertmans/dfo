/// Forward Mode Automatic Differentiation
use std::ops::{Add, Deref};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DFloat<T> {
    x: T,
    dx: T,
}

type DFloat32 = DFloat<f32>;
type DFloat64 = DFloat<f64>;

// Implementing std::ops traits

/*
macro_rules! impl_binop (
    ($trait:ident, $method:ident |$lhs_i:tt : $lhs:ty, $rhs_i:ident : &$rhs:ty| -> $out:ty $body:block)
    => {
    }
*/

macro_rules! impl_binop (
/*
    ($trait:ident, $method:ident |$lhs_i:tt : &$lhs:ty, $rhs_i:ident : &$rhs:ty| $body:block ) => (

impl<'a, A, B> $trait<&'a $rhs> for &'a DFloat<A>
where
    A: $trait<&'a B, Output = A>,
    &'a A: $trait<&'a B, Output = A>,
{
    type Output = DFloat<A>;
    fn $method(self, $rhs_i: &'a $rhs) -> Self::Output {
        let $lhs_i = self;
        $body
    }
}
);
    ($trait:ident, $method:ident |$lhs_i:tt : &$lhs:ty, $rhs_i:ident : $rhs:ty| $body:block ) => (

impl<'a, A, B> $trait<$rhs> for &'a DFloat<A>
where
    A: $trait<B, Output = A>,
    &'a A: $trait<B, Output = A>,
{
    type Output = DFloat<A>;
    fn $method(self, $rhs_i: $rhs) -> Self::Output {
        let $lhs_i = self;
        $body
    }
}
);
    ($trait:ident, $method:ident |$lhs_i:tt : $lhs:ty, $rhs_i:ident : &$rhs:ty| $body:block ) => (

impl<'a, A, B> $trait<&'a $rhs> for DFloat<A>
where
    A: $trait<&'a B, Output = A>,
{
    type Output = Self;
    fn $method(self, $rhs_i: &'a $rhs) -> Self::Output {
        let $lhs_i = self;
        $body
    }
}
);
*/
    ($trait:ident, $method:ident |$xa:tt : $_xa:ty, $dxa:tt : $_dxa:ty, $xb:tt : $_xb:ty, $dxb:tt : $_dxb:ty| $body:block ) => (

        impl_binop!($trait, $method, DFloat<A>, DFloat<B> |$xa: $_xa, $dxa: $_dxa, $xb: $_xb, $dxb: $_dxb| $body);
        impl_binop!($trait, $method, DFloat<A>, &DFloat<B> |$xa: $_xa, $dxa: $_dxa, $xb: $_xb, $dxb: $_dxb| $body);
        impl_binop!($trait, $method, &DFloat<A>, DFloat<B> |$xa: $_xa, $dxa: $_dxa, $xb: $_xb, $dxb: $_dxb| $body);
        impl_binop!($trait, $method, &DFloat<A>, &DFloat<B> |$xa: $_xa, $dxa: $_dxa, $xb: $_xb, $dxb: $_dxb| $body);

);
    ($trait:ident, $method:ident, &$lhs:ty, &$rhs:ty |$xa:tt : $_xa:ty, $dxa:tt : $_dxa:ty, $xb:tt : $_xb:ty, $dxb:tt : $_dxb:ty| $body:block ) => (

impl<'a, A, B> $trait<&'a $rhs> for &'a DFloat<A>
where
    A: $trait<B, Output = A>,
    &'a A: $trait<&'a B, Output = A>,
{
    type Output = DFloat<A>;
    fn $method(self, other: &'a $rhs) -> Self::Output {
        let ($xa, $dxa) = (&self.x, &self.dx);
        let ($xb, $dxb) = (&other.x, &other.dx);
        $body
    }
}
);
    ($trait:ident, $method:ident, &$lhs:ty, $rhs:ty |$xa:tt : $_xa:ty, $dxa:tt : $_dxa:ty, $xb:tt : $_xb:ty, $dxb:tt : $_dxb:ty| $body:block ) => (

impl<'a, A, B> $trait<$rhs> for &'a DFloat<A>
where
    A: $trait<B, Output = A>,
    &'a A: $trait<B, Output = A>,
{
    type Output = DFloat<A>;
    fn $method(self, other: $rhs) -> Self::Output {
        let ($xa, $dxa) = (&self.x, &self.dx);
        let ($xb, $dxb) = (other.x, other.dx);
        $body
    }
}
);
    ($trait:ident, $method:ident, $lhs:ty, &$rhs:ty |$xa:tt : $_xa:ty, $dxa:tt : $_dxa:ty, $xb:tt : $_xb:ty, $dxb:tt : $_dxb:ty| $body:block ) => (

impl<'a, A, B> $trait<&'a $rhs> for DFloat<A>
where
    A: $trait<B, Output = A>,
{
    type Output = DFloat<A>;
    fn $method(self, other: &'a $rhs) -> Self::Output {
        let ($xa, $dxa) = (self.x, self.dx);
        let ($xb, $dxb) = (*&other.x, *&other.dx);
        $body
    }
}
);
    ($trait:ident, $method:ident, $lhs:ty, $rhs:ty |$xa:tt : $_xa:ty, $dxa:tt : $_dxa:ty, $xb:tt : $_xb:ty, $dxb:tt : $_dxb:ty| $body:block ) => (

impl<A, B> $trait<$rhs> for DFloat<A>
where
    A: $trait<B, Output = A>,
{
    type Output = DFloat<A>;
    fn $method(self, other: $rhs) -> Self::Output {
        let ($xa, $dxa) = (self.x, self.dx);
        let ($xb, $dxb) = (other.x, other.dx);
        $body
    }
}
);

    );


impl_binop!(Add, add |xa: A, dxa: A, xb: B, dxb: B| { DFloat { x: xa + xb, dx: dxa + dxb }});
//impl_binop!(Div, add |xa: A, dxa: A, xb: B, dxb: B| { DFloat { x: xa / xb, dx: dxa + dxb }});
//impl_binop!(Mul, add |xa: A, dxa: A, xb: B, dxb: B| { DFloat { x: xa * xb, dx: dxa + dxb }});
//impl_binop!(Sub, add |xa: A, dxa: A, xb: B, dxb: B| { DFloat { x: xa - xb, dx: dxa - dxb }});
    /*
impl_binop!(Add, add, DFloat<A>, DFloat<B> |xa: A, dxa: A, xb: B, dxb: B| { DFloat { x: xa + xb, dx: dxa + dxb }});
impl_binop!(Add, add, DFloat<A>, &DFloat<B> |xa: A, dxa: A, xb: B, dxb: B| { DFloat { x: xa + xb, dx: dxa + dxb }});
impl_binop!(Add, add, &DFloat<A>, DFloat<B> |xa: A, dxa: A, xb: B, dxb: B| { DFloat { x: xa + xb, dx: dxa + dxb }});
impl_binop!(Add, add, &DFloat<A>, &DFloat<B> |xa: A, dxa: A, xb: B, dxb: B| { DFloat { x: xa + xb, dx: dxa + dxb }});
*/
/*
impl_binop!(Add, add, DFloat<A>, DFloat<B> |xa: A, dxa: A, xb: B, dxb: B| { DFloat { x: xa + xb, dx: dxa + dxb }});
impl_binop!(Add, add, DFloat<A>, DFloat<B> |xa: A, dxa: A, xb: B, dxb: B| { DFloat { x: xa + xb, dx: dxa + dxb }});
*/

/*
impl<T> Add for DFloat<T>
where
    T: Add<Output = T>,
{
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            x: self.x + other.x,
            dx: self.dx + other.dx,
        }
    }
}*/

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_add() {
        let a1 = arr1(&vec![1, 2, 3, 4]);
        let a2 = arr1(&vec![5, 6, 7, 8]);

        let x = DFloat {
            x: a1.clone(),
            dx: a2.clone(),
        };


        let y = DFloat { x: &a1, dx: &a2 };

        assert_eq!(x.clone() + &y, x.clone() + x.clone());
        //assert_eq!(&x + &y, &y + &x);
    }
}
