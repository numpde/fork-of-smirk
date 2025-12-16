"""
PyO3 panic-safety tests.

Goal: error paths in the Rust extension must never panic. They should raise
ordinary Python exceptions (TypeError/ValueError/OSError/RuntimeError, etc.).

Why subprocess probes: a Rust panic may be rethrown as a catchable Python
exception (often pyo3_runtime.PanicException). Catchable is still a bug, and
running probes in fresh subprocesses lets us reliably classify outcomes:
- abort/segfault (nonzero return code),
- PyO3 panic rethrow (PanicException/SystemError or panic markers),
- normal Python exception,
- unexpected success.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_CANDIDATE_MODULES = ("smirk.smirk", "smirk._smirk", "smirk")
_CANDIDATE_CLASSES = ("SmirkTokenizer", "SmirkTokenizerFast", "Tokenizer")

_PANIC_EXC_TYPES = {"PanicException", "SystemError"}
_PANIC_MARKERS = ("panicked at", "panicexception", "rust panic", "stack backtrace:")

_SNIPPET_TEMPLATE = r"""
import importlib
import importlib.machinery
import os
import sys
import traceback

CANDIDATE_MODULES = {candidate_modules!r}
CANDIDATE_CLASSES = {candidate_classes!r}

def _is_extension_module(m) -> bool:
    fn = getattr(m, "__file__", "") or ""
    return any(fn.endswith(suf) for suf in importlib.machinery.EXTENSION_SUFFIXES)

def _find_extension_tokenizer_class():
    for mod_name in CANDIDATE_MODULES:
        try:
            m = importlib.import_module(mod_name)
        except Exception:
            continue
        if not _is_extension_module(m):
            continue
        for cls_name in CANDIDATE_CLASSES:
            C = getattr(m, cls_name, None)
            if C is None:
                continue
            if any(hasattr(C, attr) for attr in ("train", "with_padding", "pretokenize", "encode", "save")):
                return (m, C)
    return (None, None)

m, Tok = _find_extension_tokenizer_class()
if Tok is None:
    print("NO_CLASS")
    sys.exit(3)

print("FOUND_CLASS", Tok.__module__, Tok.__name__)
print("FOUND_FILE", getattr(m, "__file__", ""))

try:
    {probe}
except BaseException as e:
    print("EXC_TYPE", type(e).__name__)
    print("EXC_MOD", type(e).__module__)
    print("EXC_STR", str(e))
    traceback.print_exc()
    sys.exit(0)

print("NO_EXCEPTION")
sys.exit(2)
"""


@dataclass(frozen=True)
class ProbeResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def exc_type(self) -> str | None:
        for line in self.stdout.splitlines():
            if line.startswith("EXC_TYPE "):
                return line.split(" ", 1)[1].strip()
        return None

    @property
    def has_class(self) -> bool:
        return "FOUND_CLASS" in self.stdout and "NO_CLASS" not in self.stdout

    @property
    def succeeded(self) -> bool:
        return self.returncode == 2 and "NO_EXCEPTION" in self.stdout

    @property
    def raised(self) -> bool:
        return self.returncode == 0 and "EXC_TYPE" in self.stdout

    @property
    def crashed(self) -> bool:
        return self.returncode not in (0, 2, 3)

    @property
    def panicked(self) -> bool:
        et = self.exc_type
        if et in _PANIC_EXC_TYPES:
            return True
        blob = (self.stdout + "\n" + self.stderr).lower()
        return any(m in blob for m in _PANIC_MARKERS)


def _run_probe(probe: str, env: dict[str, str] | None = None) -> ProbeResult:
    base_env = os.environ.copy()
    base_env.setdefault("RUST_BACKTRACE", "1")
    base_env.setdefault("PYTHONFAULTHANDLER", "1")
    base_env.setdefault("PYTHONUNBUFFERED", "1")
    if env:
        base_env.update(env)

    code = _SNIPPET_TEMPLATE.format(
        candidate_modules=_CANDIDATE_MODULES,
        candidate_classes=_CANDIDATE_CLASSES,
        probe=probe,
    )
    p = subprocess.run(
        [sys.executable, "-c", code],
        text=True,
        capture_output=True,
        env=base_env,
    )
    return ProbeResult(returncode=p.returncode, stdout=p.stdout or "", stderr=p.stderr or "")


def _dump(r: ProbeResult) -> None:
    print("=== subprocess returncode ===")
    print(r.returncode)
    print("=== subprocess stdout ===")
    print(r.stdout)
    print("=== subprocess stderr ===")
    print(r.stderr)


def _assert_probe_found_class(r: ProbeResult) -> None:
    if not r.has_class:
        _dump(r)
    assert r.has_class, "Could not locate an extension-module tokenizer class (wrong import / not built?)."


def _assert_no_crash(r: ProbeResult) -> None:
    if r.crashed or r.returncode == 3:
        _dump(r)
    assert r.returncode != 3, "Probe could not locate the extension tokenizer class."
    assert not r.crashed, "Subprocess crashed/aborted (likely segfault or abort)."


def _assert_raised_normal_exception(r: ProbeResult) -> None:
    _assert_probe_found_class(r)
    _assert_no_crash(r)
    if not r.raised or r.panicked:
        _dump(r)
    assert r.raised, "Probe unexpectedly succeeded; expected an exception."
    assert not r.panicked, f"Probe raised a panic-rethrow ({r.exc_type}); should raise a normal Python exception."


def _assert_no_panic_allow_success(r: ProbeResult) -> None:
    _assert_probe_found_class(r)
    _assert_no_crash(r)
    if r.panicked:
        _dump(r)
    assert not r.panicked, f"Probe panicked ({r.exc_type}); must never panic."


def test_with_padding_bad_type_raises_normal_exception() -> None:
    r = _run_probe('tok = Tok(); tok.with_padding(pad_id="x")')
    _assert_raised_normal_exception(r)


def test_with_truncation_bad_type_raises_normal_exception() -> None:
    r = _run_probe('tok = Tok(); tok.with_truncation(max_length="not_an_int")')
    _assert_raised_normal_exception(r)


def test_setstate_invalid_bytes_raises_normal_exception() -> None:
    r = _run_probe("tok = Tok(); tok.__setstate__(b'{\"invalid_json\":')")
    _assert_raised_normal_exception(r)


def test_train_missing_file_raises_normal_exception(tmp_path: Path) -> None:
    missing = tmp_path / "definitely_missing_corpus.txt"
    r = _run_probe(
        "tok = Tok(); tok.train(files=[os.environ['MISSING']], vocab_size=100)",
        env={"MISSING": str(missing)},
    )
    _assert_raised_normal_exception(r)


def test_save_invalid_target_raises_normal_exception(tmp_path: Path) -> None:
    parent_file = tmp_path / "not_a_dir"
    parent_file.write_text("x", encoding="utf-8")
    invalid = parent_file / "child.json"

    r = _run_probe(
        "tok = Tok(); tok.save(os.environ['TARGET'])",
        env={"TARGET": str(invalid)},
    )
    _assert_raised_normal_exception(r)


def test_train_then_pretokenize_does_not_panic(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("CCO\nC1CCCCC1\n", encoding="utf-8")

    probe = (
        "tok = Tok(); "
        "tok.train(files=[os.environ['CORPUS']], vocab_size=200, split_structure=False); "
        "tok.pretokenize('CCO')"
    )
    r = _run_probe(probe, env={"CORPUS": str(corpus)})

    # This one is a regression guard: success is acceptable, and a normal exception
    # is acceptable, but a panic is not.
    _assert_no_panic_allow_success(r)
