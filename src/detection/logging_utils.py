# -*- coding: utf-8 -*-
"""
Per-run console logging helpers

Provides a context manager that mirrors everything written to stdout and stderr
into a log file while still showing it on the console, so each experiment produces
its own self-contained log without changing the print-based reporting in the code
"""

import os
import sys
from contextlib import contextmanager


class _Tee:
    """
    Write-through stream that forwards every write to several underlying streams

    The object must behave like a genuine text stream because libraries such as
    transformers introspect stdout (for example calling isatty). Any attribute not
    defined here is delegated to the primary stream, and isatty reports False so
    colourised or carriage-return terminal output is disabled in the log file
    """

    def __init__(self, primary, *extra):
        self._primary = primary
        self._streams = (primary, *extra)

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return False                                # Plain text for both console and log file

    def __getattr__(self, name):
        return getattr(self._primary, name)         # Delegate encoding, fileno, buffer, etc.


@contextmanager
def tee_output(path: str):
    """
    Mirror stdout and stderr to a log file for the duration of the block

    Args:
        path (str): Destination log file, parent directories are created
    Returns:
        str: The log path, yielded for convenience
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    handle = open(path, "w", encoding="utf-8", buffering=1)     # Line-buffered for live tailing
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(saved_out, handle)
    sys.stderr = _Tee(saved_err, handle)
    try:
        yield path
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err
        handle.close()
