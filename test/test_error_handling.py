"""
Tests for error handling and failure modes across the two public Python
surfaces exposed by this project:

1) `smirk.SmirkTokenizerFast` (the Transformers-compatible “Fast” wrapper)
   - Uses the standard Transformers I/O and lifecycle methods such as
     `save_pretrained(...)` and initialization via `tokenizer_file=...`.
   - Does not expose the Rust/PyO3 tokenizer methods like `with_padding`,
     `from_file`, `save`, or `train`.

2) `smirk.smirk.SmirkTokenizer` (the core Rust/PyO3 tokenizer binding)
   - Exposes low-level tokenizer configuration and I/O methods such as
     `with_padding(...)`, `with_truncation(...)`, `from_file(...)`,
     `save(...)`, and `train(...)`.

This file intentionally verifies that common failure scenarios raise
Python exceptions (e.g., `OSError`, `TypeError`, `RuntimeError`) rather
than returning corrupted state or crashing silently.

Some tests allow for either a normal Python exception or a PyO3 panic
exception (if surfaced as `pyo3_runtime.PanicException`).
"""

import os
import tempfile
from importlib.resources import files

import pytest
import smirk
from smirk.smirk import SmirkTokenizer


def _panic_exc_type():
    # PyO3 panics are often surfaced as pyo3_runtime.PanicException.
    # Keep this optional so tests don't depend on the module name.
    try:
        import pyo3_runtime  # type: ignore

        return pyo3_runtime.PanicException
    except Exception:
        return None


def _unwritable_dir():
    d = tempfile.mkdtemp()
    # remove write bit (owner/group/other)
    os.chmod(d, 0o555)
    return d


def _restore_writable(path: str):
    try:
        os.chmod(path, 0o755)
    except Exception:
        pass


def test_fast_init_missing_tokenizer_file_raises_oserror():
    with pytest.raises(OSError):
        smirk.SmirkTokenizerFast(tokenizer_file="does_not_exist.json")


def test_fast_save_pretrained_unwritable_dir_raises_oserror():
    tok = smirk.SmirkTokenizerFast()
    d = _unwritable_dir()
    try:
        with pytest.raises(OSError):
            tok.save_pretrained(d)
    finally:
        _restore_writable(d)
        try:
            os.rmdir(d)
        except Exception:
            pass


def test_train_gpe_missing_file_raises_oserror():
    with pytest.raises(OSError):
        smirk.train_gpe(["missing.txt"])


def test_core_with_padding_invalid_kwargs_raises():
    vocab_file = files("smirk").joinpath("vocab_smiles.json")
    tok = SmirkTokenizer.from_vocab(str(vocab_file))

    panic_exc = _panic_exc_type()
    expected = (TypeError, RuntimeError) if panic_exc is None else (TypeError, RuntimeError, panic_exc)

    with pytest.raises(expected):
        tok.with_padding(pad_id="not an int")


def test_core_from_file_missing_raises_oserror():
    with pytest.raises(OSError):
        SmirkTokenizer.from_file("does_not_exist.json")


def test_core_save_unwritable_path_raises_oserror():
    vocab_file = files("smirk").joinpath("vocab_smiles.json")
    tok = SmirkTokenizer.from_vocab(str(vocab_file))

    d = _unwritable_dir()
    try:
        with pytest.raises(OSError):
            tok.save(os.path.join(d, "tok.json"))
    finally:
        _restore_writable(d)
        try:
            os.rmdir(d)
        except Exception:
            pass


def test_core_pretokenize_without_pretok_errors():
    vocab_file = files("smirk").joinpath("vocab_smiles.json")
    tok = SmirkTokenizer.from_vocab(str(vocab_file))

    with tempfile.NamedTemporaryFile("w", delete=False) as handle:
        handle.write("CCO\n")
        path = handle.name

    panic_exc = _panic_exc_type()
    expected = (RuntimeError,) if panic_exc is None else (RuntimeError, panic_exc)

    try:
        trained = tok.train([path], split_structure=False)
        with pytest.raises(expected):
            trained.pretokenize("CCO")
    finally:
        os.remove(path)
