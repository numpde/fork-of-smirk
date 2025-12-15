use std::io;

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::PyErr;
use tokenizers::Error as TokenizerError;

pub fn value_err(ctx: &str, msg: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(format!("{}: {}", ctx, msg))
}

pub fn type_err(ctx: &str, msg: impl std::fmt::Display) -> PyErr {
    PyTypeError::new_err(format!("{}: {}", ctx, msg))
}

pub fn io_err(ctx: &str, err: std::io::Error) -> PyErr {
    PyIOError::new_err(format!("{}: {}", ctx, err))
}

pub fn runtime_err(ctx: &str, err: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!("{}: {}", ctx, err))
}

pub fn map_tok_err(ctx: &str, err: TokenizerError) -> PyErr {
    if let Some(io_error) = err.downcast_ref::<io::Error>() {
        io_err(ctx, io::Error::new(io_error.kind(), io_error.to_string()))
    } else {
        runtime_err(ctx, err)
    }
}
