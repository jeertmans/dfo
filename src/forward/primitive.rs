//! Forward automatic differentiation implemented on primitive types.
//!
#[cfg(feature = "num-traits")]
use num_traits::{Float, Num, NumCast, One, ToPrimitive, Zero};
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
/// Automatically differentiated float.
///
/// This structure only implements interesting methods for
/// the following primitive types:
/// - [`f32`], see [`DFloat32`] type alias
/// - [`f64`], see [`DFloat64`] type alias
pub struct DFloat<T> {
    x: T,
    dx: T,
}

macro_rules! impl_from {
    ($($t:ty)*) => ($(
        impl From<$t> for DFloat<$t> {
            #[inline]
            fn from(x: $t) -> Self {
                Self { x, dx: 0 as $t }
            }
        }
        impl From<&$t> for DFloat<$t> {
            #[inline]
            fn from(x: &$t) -> Self {
                Self { x: *x, dx: 0 as $t }
            }
        }
        impl From<($t,$t)> for DFloat<$t> {
            #[inline]
            fn from(x: ($t,$t)) -> Self {
                Self { x: x.0, dx: x.1 }
            }
        }
        impl From<(&$t, &$t)> for DFloat<$t> {
            #[inline]
            fn from(x: (&$t, &$t)) -> Self {
                Self { x: *(x.0), dx: *(x.1) }
            }
        }
    )*)
}

impl_from!(f32 f64);

#[cfg(test)]
macro_rules! test_from {
    ($($t:ty)*) => ($(
        concat_idents::concat_idents!(test_name = test_from, _, $t {
            #[test]
            fn test_name() {
                let d1: DFloat<$t> = 4.0.into();
                let d2: DFloat<$t> = (&4.0).into();
                let d3: DFloat<$t> = (4.0, 8.0).into();
                let d4: DFloat<$t> = (&4.0, &8.0).into();
                assert_eq!(d1, DFloat { x: 4.0, dx: 0.0 });
                assert_eq!(d2, DFloat { x: 4.0, dx: 0.0 });
                assert_eq!(d3, DFloat { x: 4.0, dx: 8.0 });
                assert_eq!(d4, DFloat { x: 4.0, dx: 8.0 });
            }
        });
    )*)
}

#[cfg(test)]
mod from_tests {
    use super::*;
    test_from!(f32 f64);
}

macro_rules! impl_neg {
    ($($t:ty)*) => ($(
        impl Neg for DFloat<$t> {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self::Output {
                DFloat { x: -self.x, dx: -self.dx }
            }
        }
        impl Neg for &DFloat<$t> {
            type Output = DFloat<$t>;
            #[inline]
            fn neg(self) -> Self::Output {
                DFloat { x: -self.x, dx: -self.dx }
            }
        }
    )*)
}

impl_neg!(f32 f64);

#[cfg(test)]
macro_rules! test_neg {
    ($($t:ty)*) => ($(
        concat_idents::concat_idents!(test_name = test_neg, _, $t {
            #[test]
            fn test_name() {
                let d1: DFloat<$t> = 4.0.into();
                let d2: DFloat<$t> = (&4.0).into();
                assert_eq!(-d1, DFloat { x: -4.0, dx: -0.0 });
                assert_eq!(-(&d2), DFloat { x: -4.0, dx: -0.0 });
            }
        });
    )*)
}

#[cfg(test)]
mod neg_tests {
    use super::*;
    test_neg!(f32 f64);
}

#[cfg(feature = "num-traits")]
macro_rules! impl_one {
    ($($t:ty)*) => ($(
        impl One for DFloat<$t> {
            #[inline]
            fn one() -> Self::Output {
                DFloat { x: 1 as $t, dx: 0 as $t }
            }
            #[inline]
            fn is_one(&self) -> bool {
                self.x.is_one() && self.dx.is_zero()
            }
        }
    )*)
}

#[cfg(feature = "num-traits")]
impl_one!(f32 f64);

#[cfg(feature = "num-traits")]
#[cfg(test)]
macro_rules! test_one {
    ($($t:ty)*) => ($(
        concat_idents::concat_idents!(test_name = test_one, _, $t {
            #[test]
            fn test_name() {
                let d1: DFloat<$t> = (1.0, 0.0).into();
                let d2: DFloat<$t> = (2.0, 0.0).into();
                let d3: DFloat<$t> = (1.0, 0.5).into();
                let d4: DFloat<$t> = (0.0, 0.0).into();
                assert!(d1.is_one());
                assert!(!d2.is_one());
                assert!(!d3.is_one());
                assert!(!d4.is_one());
            }
        });
    )*)
}

#[cfg(feature = "num-traits")]
#[cfg(test)]
mod one_tests {
    use super::*;
    test_one!(f32 f64);
}

#[cfg(feature = "num-traits")]
macro_rules! impl_zero {
    ($($t:ty)*) => ($(
        impl Zero for DFloat<$t> {
            #[inline]
            fn zero() -> Self::Output {
                DFloat { x: 0 as $t, dx: 0 as $t }
            }
            #[inline]
            fn is_zero(&self) -> bool {
                self.x.is_zero() && self.dx.is_zero()
            }
        }
    )*)
}

#[cfg(feature = "num-traits")]
impl_zero!(f32 f64);

#[cfg(feature = "num-traits")]
#[cfg(test)]
macro_rules! test_zero {
    ($($t:ty)*) => ($(
        concat_idents::concat_idents!(test_name = test_zero, _, $t {
            #[test]
            fn test_name() {
                let d1: DFloat<$t> = (1.0, 0.0).into();
                let d2: DFloat<$t> = (2.0, 0.0).into();
                let d3: DFloat<$t> = (1.0, 0.5).into();
                let d4: DFloat<$t> = (0.0, 0.0).into();
                assert!(!d1.is_zero());
                assert!(!d2.is_zero());
                assert!(!d3.is_zero());
                assert!(d4.is_zero());
            }
        });
    )*)
}

#[cfg(feature = "num-traits")]
#[cfg(test)]
mod zero_tests {
    use super::*;
    test_zero!(f32 f64);
}

macro_rules! impl_binop {
    ($trt:ident,
     $mth:ident,
     $t:ty,
     $($asgn_trt:ident, $asgn_mth:ident,)?
     |$a:tt, $b:tt, $c:tt, $d:tt| $body:block) => {
        impl<T> $trt<T> for DFloat<$t>
        where
            T: Into<Self>,
        {
            type Output = Self;
            #[inline]
            fn $mth(self, other: T) -> Self::Output {
                let other: Self = other.into();
                let ($a, $b) = (self.x, self.dx);
                let ($c, $d) = (other.x, other.dx);
                $body
            }
        }
        impl $trt<&DFloat<$t>> for DFloat<$t> {
            type Output = Self;
            #[inline]
            fn $mth(self, other: &Self) -> Self::Output {
                let ($a, $b) = (self.x, self.dx);
                let ($c, $d) = (other.x, other.dx);
                $body
            }
        }
        impl<T> $trt<T> for &DFloat<$t>
        where
            T: Into<DFloat<$t>>
        {
            type Output = DFloat<$t>;
            #[inline]
            fn $mth(self, other: T) -> Self::Output {
                let other: DFloat<$t> = other.into();
                let ($a, $b) = (self.x, self.dx);
                let ($c, $d) = (other.x, other.dx);
                $body
            }
        }
        impl $trt<&DFloat<$t>> for &DFloat<$t> {
            type Output = DFloat<$t>;
            #[inline]
            fn $mth(self, other: &DFloat<$t>) -> Self::Output {
                let ($a, $b) = (self.x, self.dx);
                let ($c, $d) = (other.x, other.dx);
                $body
            }
        }
        $(
        impl<T> $asgn_trt<T> for DFloat<$t>
        where
            T: Into<Self>,
        {
            #[inline]
            fn $asgn_mth(&mut self, other: T) {
                let other: Self = other.into();
                let ($a, $b) = (self.x, self.dx);
                let ($c, $d) = (other.x, other.dx);
                *self = $body
            }
        }
        impl $asgn_trt<&DFloat<$t>> for DFloat<$t> {
            #[inline]
            fn $asgn_mth(&mut self, other: &Self) {
                let ($a, $b) = (self.x, self.dx);
                let ($c, $d) = (other.x, other.dx);
                *self = $body
            }
        }
        )?
    };
}

macro_rules! impl_all_binops {
    ($($t:ty)*) => ($(
        impl_binop!(Add, add, $t,
                    AddAssign, add_assign,
                    |a, da, b, db| {
                        DFloat { x: a + b, dx: da + db }
                    });
        impl_binop!(Div, div, $t,
                    DivAssign, div_assign,
                    |a, da, b, db| {
                        DFloat { x: a / b, dx: (a * db - da * b) / (b * b) }
                    });
        impl_binop!(Mul, mul, $t,
                    MulAssign, mul_assign,
                    |a, da, b, db| {
                        DFloat { x: a * b, dx: a * db + da * b }
                    });
        impl_binop!(Rem, rem, $t,
                    RemAssign, rem_assign,
                    |a, da, b, db| {
                        let div = a / b;
                        let rem = div % (1  as $t);
                        DFloat { x: a % b, dx: da - db * (div - rem) }
                    });
        impl_binop!(Sub, sub, $t,
                    SubAssign, sub_assign,
                    |a, da, b, db| {
                        DFloat { x: a - b, dx: da - db }
                    });
    )*)
}

impl_all_binops!(f32 f64);

#[cfg(test)]
macro_rules! test_binop {
    ($mth:ident $t:ty) => (
        concat_idents::concat_idents!(test_name = test_binop_, $mth, _, $t, _compiles_and_has_expected_value {
            #[test]
            fn test_name() {
                let v1: $t = 4 as $t;
                let v2: $t = 8 as $t;
                let d1: DFloat<$t> = (&v1).into();
                let d2: DFloat<$t> = (&v2).into();

                let _: DFloat<$t> = d1.clone().$mth(v2.clone());
                let _: DFloat<$t> = d1.clone().$mth(&v2);
                let _: DFloat<$t> = (&d1).$mth(v2.clone());
                let _: DFloat<$t> = (&d1).$mth(&v2);
                let _: DFloat<$t> = d1.clone().$mth(d2.clone());
                let _: DFloat<$t> = d1.clone().$mth(&d2);
                let _: DFloat<$t> = (&d1).$mth(d2.clone());
                let _: DFloat<$t> = (&d1).$mth(&d2);

                let expected: DFloat<_> = (v1.$mth(v2)).into();
                assert_eq!(expected, d1.$mth(d2));
            }
        });
    )
}

#[cfg(test)]
macro_rules! test_all_binops {
    ($($t:ty)*) => ($(
        test_binop!(add $t);
        test_binop!(div $t);
        test_binop!(mul $t);
        test_binop!(rem $t);
        test_binop!(sub $t);
    )*)
}

#[cfg(test)]
mod binop_tests {
    use super::*;
    test_all_binops!(f32 f64);
}

//impl NumOps for DFloat<f32> {}
//impl NumOps for DFloat<f64> {}

#[cfg(feature = "num-traits")]
macro_rules! impl_num {
    ($($t:ty)*) => ($(
        impl Num for DFloat<$t> {
            type FromStrRadixErr = num_traits::ParseFloatError;
            fn from_str_radix(s: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                <$t>::from_str_radix(s, radix).map(|x| x.into())
            }
        }
    )*)
}

#[cfg(feature = "num-traits")]
impl_num!(f32 f64);

#[cfg(feature = "num-traits")]
macro_rules! impl_to_primitive {
    ($($t:ty)*) => ($(
        impl ToPrimitive for DFloat<$t> {
            fn to_i64(&self) -> Option<i64> {
                None
            }
            fn to_u64(&self) -> Option<u64> {
                None
            }
        }
    )*)
}

#[cfg(feature = "num-traits")]
impl_to_primitive!(f32 f64);

#[cfg(feature = "num-traits")]
macro_rules! impl_num_cast {
    ($($t:ty)*) => ($(
        impl NumCast for DFloat<$t> {
            fn from<T: ToPrimitive>(n: T) -> Option<Self> {
                None
            }
        }
    )*)
}

#[cfg(feature = "num-traits")]
impl_num_cast!(f32 f64);

#[cfg(feature = "num-traits")]
macro_rules! impl_float {
    ($($t:ty)*) => ($(
        impl Float for DFloat<$t> {
            #[inline]
            fn nan() -> Self {
                Self { x: <$t>::nan(), dx: <$t>::nan() }
            }
            #[inline]
            fn infinity() -> Self {
                Self { x: <$t>::infinity(), dx: <$t>::infinity() }
            }
            #[inline]
            fn neg_infinity() -> Self {
                Self { x: <$t>::neg_infinity(), dx: <$t>::neg_infinity() }
            }
            #[inline]
            fn neg_zero() -> Self {
                Self { x: <$t>::neg_zero(), dx: <$t>::neg_zero() }
            }
            #[inline]
            fn min_value() -> Self {
                Self { x: <$t>::min_value(), dx: <$t>::min_value() }
            }
            #[inline]
            fn min_positive_value() -> Self {
                Self { x: <$t>::min_positive_value(), dx: <$t>::min_positive_value() }
            }
            #[inline]
            fn max_value() -> Self {
                Self { x: <$t>::max_value(), dx: <$t>::max_value() }
            }
            #[inline]
            fn is_nan(self) -> bool {
                self.x.is_nan() || self.dx.is_nan()
            }
            #[inline]
            fn is_infinite(self) -> bool {
                self.x.is_infinite() || self.dx.is_infinite()
            }
            #[inline]
            fn is_finite(self) -> bool {
                self.x.is_finite() && self.dx.is_finite()
            }
            #[inline]
            fn is_normal(self) -> bool {
                self.x.is_normal() || self.dx.is_normal()
            }
            #[inline]
            fn classify(self) -> std::num::FpCategory {
                self.x.classify()
            }
            #[inline]
            fn floor(self) -> Self {
                todo!()
            }
            #[inline]
            fn ceil(self) -> Self {
                todo!()
            }
            #[inline]
            fn round(self) -> Self {
                todo!()
            }
            #[inline]
            fn trunc(self) -> Self {
                todo!()
            }
            #[inline]
            fn fract(self) -> Self {
                Self { x: self.x.fract(), dx: self.dx.fract() }
            }
            #[inline]
            fn abs(self) -> Self {
                Self { x: self.x.abs(), dx: self.dx.abs() }
            }
            #[inline]
            fn signum(self) -> Self {
                Self { x: self.x.signum(), dx: self.dx.signum() }
            }
            #[inline]
            fn is_sign_positive(self) -> bool {
                self.x.is_sign_positive()
            }
            #[inline]
            fn is_sign_negative(self) -> bool {
                self.x.is_sign_negative()
            }
            #[inline]
            fn mul_add(self, a: Self, b: Self) ->Self {
                (self * a) + b
            }
            #[inline]
            fn recip(self) -> Self {
                Self::one() / self
            }
            #[inline]
            fn powi(self, n: i32) -> Self {
                todo!()
            }
            #[inline]
            fn powf(self, n: Self) -> Self {
                todo!()
            }
            #[inline]
            fn exp(self) -> Self {
                todo!()
            }
            #[inline]
            fn exp2(self) -> Self {
                todo!()
            }
            #[inline]
            fn sqrt(self) -> Self {
                todo!()
            }
            #[inline]
            fn ln(self) -> Self {
                todo!()
            }
            #[inline]
            fn log(self, base: Self) -> Self {
                todo!()
            }
            #[inline]
            fn log2(self) -> Self {
                todo!()
            }
            #[inline]
            fn log10(self) -> Self {
                todo!()
            }
            #[inline]
            fn max(self, other: Self) -> Self {
                todo!()
            }
            #[inline]
            fn min(self, other: Self) -> Self {
                todo!()
            }
            #[inline]
            fn abs_sub(self, other: Self) -> Self {
                todo!()
            }
            #[inline]
            fn cbrt(self) -> Self {
                todo!()
            }
            #[inline]
            fn hypot(self, other: Self) -> Self {
                todo!()
            }
            #[inline]
            fn sin(self) -> Self {
                todo!()
            }
            #[inline]
            fn cos(self) -> Self {
                todo!()
            }
            #[inline]
            fn tan(self) -> Self {
                todo!()
            }
            #[inline]
            fn asin(self) -> Self {
                todo!()
            }
            #[inline]
            fn acos(self) -> Self {
                todo!()
            }
            #[inline]
            fn atan(self) -> Self {
                todo!()
            }
            #[inline]
            fn atan2(self, other: Self) -> Self {
                todo!()
            }
            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                todo!()
            }
            #[inline]
            fn exp_m1(self) -> Self {
                todo!()
            }
            #[inline]
            fn ln_1p(self) -> Self {
                todo!()
            }
            #[inline]
            fn sinh(self) -> Self {
                todo!()
            }
            #[inline]
            fn cosh(self) -> Self {
                todo!()
            }
            #[inline]
            fn tanh(self) -> Self {
                todo!()
            }
            #[inline]
            fn asinh(self) -> Self {
                todo!()
            }
            #[inline]
            fn acosh(self) -> Self {
                todo!()
            }
            #[inline]
            fn atanh(self) -> Self {
                todo!()
            }
            #[inline]
            fn integer_decode(self) -> (u64, i16, i8) {
                self.x.integer_decode()
            }
        }
    )*)
}

#[cfg(feature = "num-traits")]
impl_float!(f32 f64);

/// Automatically differentiated float using [`f32`] as base type.
pub type DFloat32 = DFloat<f32>;
/// Automatically differentiated float using [`f64`] as base type.
pub type DFloat64 = DFloat<f64>;
