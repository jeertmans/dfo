#![deny(missing_docs)]
#![deny(missing_debug_implementations)]
#![warn(clippy::must_use_candidate)]
#![allow(clippy::doc_markdown, clippy::module_name_repetitions)]
#![cfg_attr(docsrs, feature(doc_auto_cfg))]
#![doc = include_str!("../README.md")]
pub mod forward;
pub mod backward;
