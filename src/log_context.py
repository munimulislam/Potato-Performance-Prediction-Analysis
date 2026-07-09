"""
@File - log_context.py
@Author - MdMunimul.Islam@teagasc.ie
@Created - 09/07/2026
"""

from contextvars import ContextVar

_run_id: ContextVar[str] = ContextVar("run_id", default="-")


def set_run_id(run_id: str) -> None:
    _run_id.set(run_id)


def get_run_id() -> str:
    return _run_id.get()
