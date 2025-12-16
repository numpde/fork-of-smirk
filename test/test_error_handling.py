import os
import tempfile

import pytest

import smirk


@pytest.mark.parametrize("kwargs,expected", [({"pad_id": "not an int"}, TypeError)])
def test_with_padding_invalid_kwargs(kwargs, expected):
    tok = smirk.SmirkTokenizerFast()
    with pytest.raises(expected):
        tok.with_padding(**kwargs)


def test_from_file_missing_raises_ioerror():
    with pytest.raises(OSError):
        smirk.SmirkTokenizerFast.from_file("does_not_exist.json")


def test_save_unwritable_path_raises_ioerror():
    tok = smirk.SmirkTokenizerFast()
    with pytest.raises(OSError):
        tok.save("/root/forbidden.json")


def test_train_missing_file_raises_ioerror():
    tok = smirk.SmirkTokenizerFast()
    with pytest.raises(OSError):
        tok.train(["missing.txt"])


def test_pretokenize_without_pretok_errors():
    tok = smirk.SmirkTokenizerFast()
    with tempfile.NamedTemporaryFile("w", delete=False) as handle:
        handle.write("CCO")
        path = handle.name
    try:
        trained = tok.train([path])
        with pytest.raises(RuntimeError):
            trained.pretokenize("CCO")
    finally:
        os.remove(path)
