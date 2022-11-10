//use num_traits::Float;
use forward_ref_generic::{forward_ref_binop, forward_ref_op_assign};
use num_traits::{Zero, One};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};
use std::fmt;

#[derive(Clone, Copy)]
pub struct DFloat<T> {
    x: T,
    dx: T,
}

use DFloat as DF;

impl<T: One> DFloat<T> {
    fn var(x: T) -> Self {
        Self { x, dx: T::one() }
    }
}

type DF32 = DFloat<f32>;

impl<T: Zero> From<T> for DFloat<T> {
    fn from(x: T) -> Self {
        Self { x, dx: T::zero() }
    }
}

impl<T: fmt::Debug> fmt::Debug for DFloat<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Point")
         .field("x", &self.x)
         .field("dx", &self.dx)
         .finish()
    }
}

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
}

forward_ref_binop! {
    [T]
    impl Add, add for DFloat<T>
    where T: Copy + Add<Output = T>
}

impl<T> AddAssign for DFloat<T>
where
    T: Copy + Add<Output = T>,
{
    #[inline]
    fn add_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x + other.x,
            dx: self.dx + other.dx,
        };
    }
}

forward_ref_op_assign! {
    [T]
    impl AddAssign, add_assign for DFloat<T>
    where T: Copy + Add<Output = T>
}

impl<T> Div for DFloat<T>
where
    T: Copy + Div<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self {
        Self {
            x: self.x / other.x,
            dx: (self.dx * other.x - other.dx * self.x) / (other.x * other.x),
        }
    }
}

forward_ref_binop! {
    [T]
    impl Div, div for DFloat<T>
    where T: Copy + Div<Output = T> + Mul<Output = T> + Sub<Output = T>
}

impl<T> DivAssign for DFloat<T>
where
    T: Copy + Div<Output = T> + Mul<Output = T> + Sub<Output = T>,
{
    #[inline]
    fn div_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x / other.x,
            dx: (self.dx * other.x - other.dx * self.x) / (other.x * other.x),
        };
    }
}

forward_ref_op_assign! {
    [T]
    impl DivAssign, div_assign for DFloat<T>
    where T: Copy + Div<Output = T> + Mul<Output = T> + Sub<Output = T>
}

impl<T> Mul for DFloat<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        Self {
            x: self.x * other.x,
            dx: self.dx * other.x + other.dx * self.x,
        }
    }
}

forward_ref_binop! {
    [T]
    impl Mul, mul for DFloat<T>
    where T: Copy + Add<Output = T> + Mul<Output = T>
}

impl<T> MulAssign for DFloat<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T>,
{
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x * other.x,
            dx: self.dx * other.x + other.dx * self.x,
        };
    }
}

forward_ref_op_assign! {
    [T]
    impl MulAssign, mul_assign for DFloat<T>
    where T: Copy + Add<Output = T> + Mul<Output = T>
}

impl<T: Sub<Output = T>> Sub for DFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            dx: self.dx - other.dx,
        }
    }
}

forward_ref_binop! {
    [T]
    impl Sub, sub for DFloat<T>
    where T: Copy + Sub<Output = T>
}

impl<T: Copy + Sub<Output = T>> SubAssign for DFloat<T> {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        *self = Self {
            x: self.x - other.x,
            dx: self.dx - other.dx,
        };
    }
}

forward_ref_op_assign! {
    [T]
    impl SubAssign, sub_assign for DFloat<T>
    where T: Copy + Sub<Output = T>
}

pub fn f(x: DF32, y: DF32) -> DF32 {
    (x + x * x) + y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = f(DF::var(2.0), 2.0.into());
        println!("{:?}", result);
        assert_eq!(3, 4);
    }
}
