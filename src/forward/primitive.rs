//! Forward automatic differentiation implemented on primitive types.
//!
//! [`DFloat32`] and [`DFloat64`] both implement, for their respective primitive type,
//! [`f32`] and [`f64`], automatic differentation on each numerical operation they
//! perform. E.g., addition, multiplication, division, and so one.
//!
//! # Note
//!
//! Using [`DFloat`] is useless, as does not implement anything
//! interesting but for types [`f32`] and [`f64`], which are aliased to [`DFloat32`]
//! and [`DFloat64`]. For other types, use generic module.
//! # Examples
//!
//! Using closures, computing value and derivatives become straightforward.
//!
//! ```
//! # use dfo::forward::primitive::*;
//! let f = |x| { x * x + 1.0 - x };
//! let df = |x| { 2.0 * x - 1.0 };
//! let x = DFloat32::var(4.0);
//!
//! assert_eq!(f(x).deriv(), df(x).value());
//! ```
//!
//! It is also possible to use generatic types to re-use already existing code.
//!
//! ```
//! # #[cfg(feature = "feature")] {
//! # use dfo::forward::primitive::*;
//! use num_traits::Float;
//!
//! fn f<T: Float + From<f32>>(x: T) -> T {
//!     let one: T = 1.0.into();
//!     x * x + one - x
//! }
//!
//! fn df<T: Float + From<f32>>(x: T) -> T {
//!     let (one, two): (T, T) = (1.0.into(), 2.0.into());
//!     two * x - one
//! }
//!
//! fn main() {
//!     let x = DFloat32::var(4.0);
//!
//!     assert_eq!(f(x).deriv(), df(x).value());
//! }
//! # }
//! ```
//!
//! If you want to work with arrays, e.g., with `ndarray`, you can do so!
//!
//! ```
//! use dfo::forward::primitive::*;
//! use ndarray::{azip, Array1};
//!
//! let x: Array1<DFloat32> =
//!     Array1::from_iter([1., 2., 3., 4.].iter().map(|x| DFloat32::var(*x)));
//!
//! let y = &x * &x * 3.0;
//!
//! azip!((&a in &y, &b in &x) assert_eq!(*a.deriv(), 6.0 * b.value()));
//! ```
pub use super::traits::Differentiable;
#[cfg(feature = "num-traits")]
use num_traits::{Float, FloatConst, FromPrimitive, Num, NumCast, One, ToPrimitive, Zero};
#[cfg(feature = "serde")]
use serde::ser::{Serialize, Serializer, SerializeStruct};
#[cfg(feature = "serde")]
use serde::de::{self, Deserialize, Deserializer, Visitor, SeqAccess, MapAccess};
#[cfg(feature = "serde")]
use std::fmt;
//use serde::{Deserialize, Serialize};
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

#[cfg(feature = "serde")]
macro_rules! impl_serde {
    ($($t:ty)*) => ($(
        impl Serialize for DFloat<$t> {
            fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
            where
                S: Serializer,
            {
                let mut state = serializer.serialize_struct("DFloat", 2)?;
                state.serialize_field("x", &self.x)?;
                state.serialize_field("dx", &self.dx)?;
                state.end()
            }
        }

        impl<'de> Deserialize<'de> for DFloat<$t> {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                enum Field { Value, Deriv }
        
                impl<'de> Deserialize<'de> for Field {
                    fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
                    where
                        D: Deserializer<'de>,
                    {
                        struct FieldVisitor;
        
                        impl<'de> Visitor<'de> for FieldVisitor {
                            type Value = Field;
        
                            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                                formatter.write_str("`x` or `dx`")
                            }
        
                            fn visit_str<E>(self, value: &str) -> Result<Field, E>
                            where
                                E: de::Error,
                            {
                                match value {
                                    "x" => Ok(Field::Value),
                                    "dx" => Ok(Field::Deriv),
                                    _ => Err(de::Error::unknown_field(value, FIELDS)),
                                }
                            }
                        }
        
                        deserializer.deserialize_identifier(FieldVisitor)
                    }
                }
        
                struct DFloatVisitor;
        
                impl<'de> Visitor<'de> for DFloatVisitor {
                    type Value = DFloat<$t>;
        
                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("struct DFloat")
                    }
        
                    fn visit_seq<V>(self, mut seq: V) -> Result<DFloat<$t>, V::Error>
                    where
                        V: SeqAccess<'de>,
                    {
                        let x = seq.next_element()?
                            .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                        let dx = seq.next_element()?
                            .ok_or_else(|| de::Error::invalid_length(1, &self))?;
                        Ok(DFloat { x, dx })
                    }
        
                    fn visit_map<V>(self, mut map: V) -> Result<DFloat<$t>, V::Error>
                    where
                        V: MapAccess<'de>,
                    {
                        let mut x = None;
                        let mut dx = None;
                        while let Some(key) = map.next_key()? {
                            match key {
                                Field::Value => {
                                    if x.is_some() {
                                        return Err(de::Error::duplicate_field("x"));
                                    }
                                    x = Some(map.next_value()?);
                                }
                                Field::Deriv => {
                                    if dx.is_some() {
                                        return Err(de::Error::duplicate_field("dx"));
                                    }
                                    dx = Some(map.next_value()?);
                                }
                            }
                        }
                        let x = x.ok_or_else(|| de::Error::missing_field("x"))?;
                        let dx = dx.ok_or_else(|| de::Error::missing_field("dx"))?;
                        Ok(DFloat { x, dx })
                    }
                }
        
                const FIELDS: &'static [&'static str] = &["x", "dx"];
                deserializer.deserialize_struct("struct DFloat", FIELDS, DFloatVisitor)
            }
        }
    )*)
}

#[cfg(feature = "serde")]
impl_serde!(f32 f64);

#[cfg(all(test, feature = "serde"))]
macro_rules! test_serde {
    ($($t:ty)*) => ($(
        concat_idents::concat_idents!(test_name = test_serde, _, $t {
            #[test]
            fn test_name() {
                let d1: DFloat<$t> = DFloat { x: 3.14, dx: 7.0 };
                let serialized = serde_json::to_string(&d1).unwrap();

                println!("{:?}", serialized);

                let deserialized: DFloat<$t> = serde_json::from_str(&serialized).unwrap();

                assert_eq!(d1, deserialized);
            }
        });
    )*)
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;
    test_serde!(f32 f64);
}

macro_rules! impl_differentiable {
    ($($t:ty)*) => ($(
        impl Differentiable for DFloat<$t> {
            type Inner = $t;
            #[inline]
            fn var(x: Self::Inner) -> Self {
                Self { x, dx: 1 as $t }
            }
            #[inline]
            fn cst(x: Self::Inner) -> Self {
                Self { x, dx: 0 as $t }
            }
            #[inline]
            fn value(&self) -> &Self::Inner {
                &self.x
            }
            #[inline]
            fn deriv(&self) -> &Self::Inner {
                &self.dx
            }
            #[inline]
            fn from_tuple(x: Self::Inner, dx: Self::Inner) -> Self {
                Self { x, dx }
            }
            #[inline]
            fn into_tuple(self) -> (Self::Inner, Self::Inner) {
                (self.x, self.dx )
            }
        }
    )*)
}

impl_differentiable!(f32 f64);

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
        impl From<DFloat<$t>> for ($t, $t) {
            #[inline]
            fn from(x: DFloat<$t>) -> ($t, $t) {
                (x.x, x.dx)
            }
        }
        impl From<&DFloat<$t>> for ($t, $t) {
            #[inline]
            fn from(x: &DFloat<$t>) -> ($t, $t) {
                (*(&x.x), *(&x.dx))
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
                let t1: ($t, $t) = d3.clone().into();
                let t2: ($t, $t) = (&d3).into();
                assert_eq!(d1, DFloat { x: 4.0, dx: 0.0 });
                assert_eq!(d2, DFloat { x: 4.0, dx: 0.0 });
                assert_eq!(d3, DFloat { x: 4.0, dx: 8.0 });
                assert_eq!(d4, DFloat { x: 4.0, dx: 8.0 });
                assert_eq!(t1, (4.0, 8.0));
                assert_eq!(t2, (4.0, 8.0));
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

#[cfg(all(test, feature = "num-traits"))]
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

#[cfg(all(test, feature = "num-traits"))]
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

#[cfg(all(test, feature = "num-traits"))]
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

#[cfg(all(test, feature = "num-traits"))]
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
        impl $trt<DFloat<$t>> for $t {
            type Output = DFloat<$t>;
            #[inline]
            fn $mth(self, other: DFloat<$t>) -> Self::Output {
                let ($a, $b) = (self, 0 as $t);
                let ($c, $d) = (other.x, other.dx);
                $body
            }
        }
        impl $trt<&DFloat<$t>> for $t {
            type Output = DFloat<$t>;
            #[inline]
            fn $mth(self, other: &DFloat<$t>) -> Self::Output {
                let ($a, $b) = (self, 0 as $t);
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
                        DFloat { x: a / b, dx: (da * b - a * db) / (b * b) }
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
macro_rules! impl_from_primitive {
    ($($t:ty)*) => ($(
        impl FromPrimitive for DFloat<$t> {
            fn from_i64(n: i64) -> Option<Self> {
                <$t>::from_i64(n).map(|x| x.into())
            }
            fn from_u64(n: u64) -> Option<Self> {
                <$t>::from_u64(n).map(|x| x.into())
            }
        }
    )*)
}

#[cfg(feature = "num-traits")]
impl_from_primitive!(f32 f64);

#[cfg(feature = "num-traits")]
macro_rules! impl_to_primitive {
    ($($t:ty)*) => ($(
        impl ToPrimitive for DFloat<$t> {
            fn to_i64(&self) -> Option<i64> {
                self.x.to_i64()
            }
            fn to_u64(&self) -> Option<u64> {
                self.x.to_u64()
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
                NumCast::from(n).map(|x| Self::cst(x))
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
                Self::cst(<$t>::nan())
            }
            #[inline]
            fn infinity() -> Self {
                Self::cst(<$t>::infinity())
            }
            #[inline]
            fn neg_infinity() -> Self {
                Self::cst(<$t>::neg_infinity())
            }
            #[inline]
            fn neg_zero() -> Self {
                Self::cst(<$t>::neg_zero())
            }
            #[inline]
            fn min_value() -> Self {
                Self::cst(<$t>::min_value())
            }
            #[inline]
            fn min_positive_value() -> Self {
                Self::cst(<$t>::min_positive_value())
            }
            #[inline]
            fn max_value() -> Self {
                Self::cst(<$t>::max_value())
            }
            #[inline]
            fn is_nan(self) -> bool {
                self.x.is_nan()
            }
            #[inline]
            fn is_infinite(self) -> bool {
                self.x.is_infinite()
            }
            #[inline]
            fn is_finite(self) -> bool {
                self.x.is_finite()
            }
            #[inline]
            fn is_normal(self) -> bool {
                self.x.is_normal()
            }
            #[inline]
            fn classify(self) -> std::num::FpCategory {
                self.x.classify()
            }
            #[inline]
            fn floor(self) -> Self {
                Self::cst(self.x.floor())
            }
            #[inline]
            fn ceil(self) -> Self {
                Self::cst(self.x.ceil())
            }
            #[inline]
            fn round(self) -> Self {
                Self::cst(self.x.round())
            }
            #[inline]
            fn trunc(self) -> Self {
                Self::cst(self.x.trunc())
            }
            #[inline]
            fn fract(self) -> Self {
                Self::cst(self.x.fract())
            }
            #[inline]
            fn abs(self) -> Self {
                if self.x.is_sign_positive() {
                    self
                } else if self.x.is_sign_negative() {
                    -self
                } else {
                    Self { x: self.x, dx: <$t>::nan() }
                }
            }
            #[inline]
            fn signum(self) -> Self {
                Self::cst(self.x.signum())
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
                let x = self.x.mul_add(a.x, b.x);
                let dx = self.x.mul_add(a.dx, b.dx) + self.dx * a.x;
                Self { x, dx }
            }
            #[inline]
            fn recip(self) -> Self {
                let x = self.x.recip();
                let dx = self.dx * x * x;
                Self { x, dx }
            }
            #[inline]
            fn powi(self, n: i32) -> Self {
                let x = self.x.powi(n);
                let dx = (n as $t) * self.x.powi(n - 1) * self.dx;
                Self { x, dx }
            }
            #[inline]
            fn powf(self, n: Self) -> Self {
                let x = self.x.powf(n.x);
                let dx = self.x.powf(n.x - (1 as $t)) * n.x.mul_add(self.dx, self.x * self.x.ln() * n.dx);
                Self { x, dx }
            }
            #[inline]
            fn sqrt(self) -> Self {
                let x = self.x.sqrt();
                let dx = - (0.5 as $t) * self.dx * x.recip();
                Self { x, dx }
            }
            #[inline]
            fn exp(self) -> Self {
                let x = self.x.exp();
                let dx = self.dx * x;
                Self { x, dx }
            }
            #[inline]
            fn exp2(self) -> Self {
                let x = self.x.exp2();
                let dx = self.dx * x * <$t>::LN_2();
                Self { x, dx }
            }
            #[inline]
            fn ln(self) -> Self {
                let x = self.x.ln();
                let dx = self.dx * self.x.recip();
                Self { x, dx }
            }
            #[inline]
            fn log(self, base: Self) -> Self {
                self.ln() / base.ln()
            }
            #[inline]
            fn log2(self) -> Self {
                let x = self.x.log2();
                let dx = self.dx * (<$t>::LN_2() * self.x).recip();
                Self { x, dx }
            }
            #[inline]
            fn log10(self) -> Self {
                let x = self.x.log10();
                let dx = self.dx * (<$t>::LN_10() * self.x).recip();
                Self { x, dx }
            }
            #[inline]
            fn max(self, other: Self) -> Self {
                if self.x > other.x {
                    self
                } else if self.x < other.x {
                    other
                } else {
                    Self { x: self.x, dx: <$t>::nan() }
                }
            }
            #[inline]
            fn min(self, other: Self) -> Self {
                if self.x < other.x {
                    self
                } else if self.x > other.x {
                    other
                } else {
                    Self { x: self.x, dx: <$t>::nan() }
                }
            }
            #[inline]
            fn abs_sub(self, other: Self) -> Self {
                (self - other).abs()
            }
            #[inline]
            fn cbrt(self) -> Self {
                let x = self.x.cbrt();
                let dx = - self.dx * (x * x * (3 as $t)).recip();
                Self { x, dx }
            }
            #[inline]
            fn hypot(self, other: Self) -> Self {
                let x = self.x.hypot(other.x);
                let dx = self.x.mul_add(self.dx, other.x * other.dx) * x.recip();
                Self { x, dx }
            }
            #[inline]
            fn sin(self) -> Self {
                let (x, cos) = self.x.sin_cos();
                let dx = self.dx * cos;
                Self { x, dx }
            }
            #[inline]
            fn cos(self) -> Self {
                let (sin, x) = self.x.sin_cos();
                let dx = - self.dx * sin;
                Self { x, dx }
            }
            #[inline]
            fn tan(self) -> Self {
                let (sin, cos) = self.x.sin_cos();
                let _cos = cos.recip();
                let dx = self.dx * _cos * _cos;
                Self { x: sin * _cos, dx }
            }
            #[inline]
            fn asin(self) -> Self {
                let x = self.x.asin();
                let dx = self.dx * ((1 as $t) - self.x * self.x).sqrt().recip();
                Self { x, dx }
            }
            #[inline]
            fn acos(self) -> Self {
                let x = self.x.acos();
                let dx = - self.dx * ((1 as $t) - self.x * self.x).sqrt().recip();
                Self { x, dx }
            }
            #[inline]
            fn atan(self) -> Self {
                let x = self.x.atan();
                let dx = self.dx * ((1 as $t) + self.x * self.x).recip();
                Self { x, dx }
            }
            #[inline]
            fn atan2(self, other: Self) -> Self {
                let x = self.x.atan2(other.x);
                let num = other.x.mul_add(self.dx, - other.dx * self.x);
                let den = self.x.mul_add(self.x, other.x * other.x);
                let dx = num * den.recip();
                Self { x, dx }
            }
            #[inline]
            fn sin_cos(self) -> (Self, Self) {
                let (sin, cos) = self.x.sin_cos();
                (Self { x: sin, dx: self.dx * cos }, Self { x: cos, dx: -self.dx * sin })
            }
            #[inline]
            fn exp_m1(self) -> Self {
                let x = self.x.exp_m1();
                let dx = self.dx * x;
                Self { x, dx }
            }
            #[inline]
            fn ln_1p(self) -> Self {
                let x = self.x.ln_1p();
                let dx = self.dx * ((1 as $t) + self.x).recip();
                Self { x, dx }
            }
            #[inline]
            fn sinh(self) -> Self {
                let x = self.x.sinh();
                let dx = self.dx * self.x.cosh();
                Self { x, dx }
            }
            #[inline]
            fn cosh(self) -> Self {
                let x = self.x.cosh();
                let dx = self.dx * self.x.sinh();
                Self { x, dx }
            }
            #[inline]
            fn tanh(self) -> Self {
                let x = self.x.tanh();
                let cosh = self.x.cosh();
                let dx = self.dx * (cosh * cosh).recip();
                Self { x, dx }
            }
            #[inline]
            fn asinh(self) -> Self {
                let x = self.x.asinh();
                let dx = self.dx * (self.x * self.x + (1 as $t)).sqrt().recip();
                Self { x, dx }
            }
            #[inline]
            fn acosh(self) -> Self {
                let x = self.x.acosh();
                let dx = self.dx * ((self.x - (1 as $t)).sqrt() * (self.x + (1 as $t)).sqrt()).recip();
                Self { x, dx }
            }
            #[inline]
            fn atanh(self) -> Self {
                let x = self.x.atanh();
                let dx = self.dx * ((1 as $t) - self.x * self.x).recip();
                Self { x, dx }
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

#[cfg(feature = "num-traits")]
macro_rules! forward_const_impl {
    ($t:ty, $($mth:ident ,)*) => ($(
        #[inline]
        fn $mth() -> Self {
            Self::cst(<$t>::$mth())
        }
    )*);
}

#[cfg(feature = "num-traits")]
macro_rules! impl_float_const {
    ($($t:ty)*) => ($(
        impl FloatConst for DFloat<$t> {
            forward_const_impl!(
                $t,
                E,
                FRAC_1_PI,
                FRAC_1_SQRT_2,
                FRAC_2_PI,
                FRAC_2_SQRT_PI,
                FRAC_PI_2,
                FRAC_PI_3,
                FRAC_PI_4,
                FRAC_PI_6,
                FRAC_PI_8,
                LN_10,
                LN_2,
                LOG10_E,
                LOG2_E,
                PI,
                SQRT_2,
            );
        }
    )*);
}

#[cfg(feature = "num-traits")]
impl_float_const!(f32 f64);

#[cfg(test)]
macro_rules! test_value_and_deriv {
    ($mth:ident, $t:ty, $var:literal $(, $vars:literal)*, |$x_var:tt $(, $o_vars:tt : $t_vars:ty)*| $body:block $(#[$attr:meta])*) => (
        concat_idents::concat_idents!(test_name = test_method_, $mth, _, $t, _has_expected_value_and_derivative {
            #[test]
            #[allow(unused_variables)]
            #[allow(unused_parens)]
            $(#[$attr])*
            fn test_name() {
                let v = DFloat::<$t>::var($var);
                let got: DFloat<$t> = DFloat::<$t>::$mth(
                    v $(, std::convert::Into::<$t_vars>::into($vars))*
                );
                let expected = {
                    let ($x_var $(, $o_vars)*) = ($var $(, $vars)*);
                    $body
                };
                assert_eq!(*got.value(), <$t>::$mth($var $(, $vars)*));
                assert_eq!(*got.deriv(), expected);
            }
        });
    )
}

#[cfg(test)]
macro_rules! test_all_ops {
    ($($t:ty)*) => ($(
        test_value_and_deriv!(add, $t, 4.0, 8.0,
                              |x, y: $t| {
                                  1 as $t
                              });
        test_value_and_deriv!(div, $t, 4.0, 8.0,
                              |x, y: $t| {
                                  (1.0 / y) as $t
                              });
        test_value_and_deriv!(mul, $t, 4.0, 8.0,
                              |x, y: $t| {
                                  y as $t
                              });
        test_value_and_deriv!(sub, $t, 4.0, 8.0,
                              |x, y: $t| {
                                  1 as $t
                              });
    )*)
}

#[cfg(all(test, feature = "num-traits"))]
macro_rules! test_all_float {
    ($($t:ty)*) => ($(
        test_value_and_deriv!(floor, $t, 4.0,
                              |x| {
                                  0 as $t
                              });
        test_value_and_deriv!(ceil, $t, 4.0,
                              |x| {
                                  0 as $t
                              });
        test_value_and_deriv!(round, $t, 4.0,
                              |x| {
                                  0 as $t
                              });
        test_value_and_deriv!(trunc, $t, 4.0,
                              |x| {
                                  0 as $t
                              });
        test_value_and_deriv!(fract, $t, 4.0,
                              |x| {
                                  0 as $t
                              });
        test_value_and_deriv!(abs, $t, -4.0,
                              |x| {
                                  x.signum() as $t
                              });
        test_value_and_deriv!(signum, $t, 4.0,
                              |x| {
                                  0 as $t
                              });
        test_value_and_deriv!(mul_add, $t, 4.0, 8.0, 7.0,
                              |x, y: DFloat<$t>, z: DFloat<$t>| {
                                  y as $t
                              });
        test_value_and_deriv!(recip, $t, 2.0,
                              |x| {
                                  (1.0 / (x * x)) as $t
                              });
        test_value_and_deriv!(powi, $t, 4.0, 3,
                              |x, n: i32| {
                                  (n as $t) * x.powi(n - 1)
                              });
        test_value_and_deriv!(powf, $t, 4.0, 3.0,
                              |x, n: DFloat<$t>| {
                                  x.powf(n) * n / x
                              });
        test_value_and_deriv!(sqrt, $t, 4.0,
                              |x| {
                                  - 0.5 * x.sqrt().recip()
                              });
        test_value_and_deriv!(exp, $t, 4.0,
                              |x| {
                                  x.exp()
                              });
        test_value_and_deriv!(exp2, $t, 4.0,
                              |x| {
                                  x.exp2() * <$t>::LN_2()
                              });
        test_value_and_deriv!(ln, $t, 4.0,
                              |x| {
                                  x.recip()
                              });
        test_value_and_deriv!(log, $t, 4.0, 17.0,
                              |x, y: DFloat<$t>| {
                                  (x * y.ln()).recip()
                              });
        test_value_and_deriv!(log2, $t, 4.0,
                              |x| {
                                  (x * <$t>::LN_2()).recip()
                              });
        test_value_and_deriv!(log10, $t, 4.0,
                              |x| {
                                  (x * <$t>::LN_10()).recip()
                              });
        test_value_and_deriv!(max, $t, 4.0, 2.0,
                              |x, y: DFloat<$t>| {
                                  1 as $t
                              });
        test_value_and_deriv!(min, $t, 4.0, 2.0,
                              |x, y: DFloat<$t>| {
                                  0 as $t
                              });
        test_value_and_deriv!(abs_sub, $t, 4.0, 2.0,
                              |x, y: DFloat<$t>| {
                                  1 as $t
                              }
                              #[allow(deprecated)]);
        test_value_and_deriv!(cbrt, $t, 4.0,
                              |x| {
                                  - (3.0 * x.cbrt().powi(2)).recip()
                              });
        test_value_and_deriv!(hypot, $t, 4.0, 8.0,
                              |x, y: DFloat<$t>| {
                                  x * x.hypot(y).recip()
                              });
        test_value_and_deriv!(sin, $t, 4.0,
                              |x| {
                                  x.cos()
                              });
        test_value_and_deriv!(cos, $t, 4.0,
                              |x| {
                                  -x.sin()
                              });
        test_value_and_deriv!(tan, $t, 4.0,
                              |x| {
                                  x.cos().powi(2).recip()
                              });
        test_value_and_deriv!(asin, $t, 0.5,
                              |x| {
                                  ((1 as $t) - x * x).sqrt().recip()
                              });
        test_value_and_deriv!(acos, $t, 0.5,
                              |x| {
                                  -((1 as $t) - x * x).sqrt().recip()
                              });
        test_value_and_deriv!(atan, $t, 4.0,
                              |x| {
                                  ((1 as $t) + x * x).recip()
                              });
        test_value_and_deriv!(atan2, $t, 4.0, 8.0,
                              |x, y: DFloat<$t>| {
                                  y * (x * x + y * y).recip()
                              });
        // sin_cos is untested since the macro does not handle returning tuples
        test_value_and_deriv!(exp_m1, $t, 4.0,
                              |x| {
                                  x.exp_m1()
                              });
        test_value_and_deriv!(ln_1p, $t, 4.0,
                              |x| {
                                  ((1 as $t) + x).recip()
                              });
        test_value_and_deriv!(sinh, $t, 4.0,
                              |x| {
                                  x.cosh()
                              });
        test_value_and_deriv!(cosh, $t, 4.0,
                              |x| {
                                  x.sinh()
                              });
        test_value_and_deriv!(tanh, $t, 4.0,
                              |x| {
                                  x.cosh().powi(2).recip()
                              });
        test_value_and_deriv!(asinh, $t, 4.0,
                              |x| {
                                  ((1 as $t) + x.powi(2)).sqrt().recip()
                              });
        test_value_and_deriv!(acosh, $t, 4.0,
                              |x| {
                                  ((x - (1 as $t)).sqrt() * (x + (1 as $t)).sqrt()).recip()
                              });
        test_value_and_deriv!(atanh, $t, 0.5,
                              |x| {
                                  ((1 as $t) - x.powi(2)).recip()
                              });
    )*)
}

#[cfg(test)]
mod ops_tests {
    use super::*;
    test_all_ops!(f32 f64);
}

#[cfg(all(test, feature = "num-traits"))]
mod test_float {
    use super::*;
    test_all_float!(f32 f64);
}

#[cfg(all(test, feature = "num-traits"))]
macro_rules! test_function {
    ($name:ident, $t:ty $(, $vars:literal)*, |$f_var:tt| $body_f:block, |$df_var:tt| $body_df:block) => (
        concat_idents::concat_idents!(test_name = test_function_, $name, _, $t, _has_expected_derivative {
            #[test]
            #[allow(unused_variables)]
            #[allow(unused_parens)]
            fn test_name() {
                $(
                    let v = DFloat::<$t>::var($vars);
                    let got = *{
                        let $f_var = v;
                        $body_f
                    }.deriv();
                    let expected = {
                        let $df_var = $vars;
                        $body_df
                    };
                    assert!((got - expected).abs() <= (1e-5 as $t), "Got: {:?} is not close enought to expected: {:?}, for input variable x = {:?}", got, expected, $vars);
                )*
            }
        });
    )
}

#[cfg(all(test, feature = "num-traits"))]
macro_rules! test_all_functions {
    ($($t:ty)*) => ($(
            test_function!(poly2, $t, -10., 0., 5.,
                          |x| { (3 as $t) * x.powi(2) + x },
                          |x| { (6 as $t) * x + (1 as $t) }
                          );
            test_function!(polyx, $t, 2., 0.1, 5.,
                          |x| { x.powf((2 as $t) * x) },
                          |x| { (2 as $t) * x.powf((2 as $t) * x) * (x.ln() + (1 as $t)) }
                          );
            test_function!(recip1, $t, -10., 0., 5.,
                          |x| { x.exp().ln() },
                          |x| { 1 as $t }
                          );
            test_function!(recip2, $t, 0.5, 0.1, 1.5,
                          |x| { x.cos().acos() },
                          |x| { 1 as $t }
                          );
            test_function!(recip3, $t, -0.1, 0., 0.1,
                          |x| { x.sin().asin().exp().ln() },
                          |x| { 1 as $t }
                          );
    )*)
}

#[cfg(all(test, feature = "num-traits"))]
mod test_functions {
    use super::*;
    test_all_functions!(f32 f64);
}

/// Automatically differentiated float using [`f32`] as base type.
pub type DFloat32 = DFloat<f32>;
/// Automatically differentiated float using [`f64`] as base type.
pub type DFloat64 = DFloat<f64>;
