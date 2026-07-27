"""Execution backends for in-process and native CLI harnesses."""

from .native_cli import NativeCliInvocation, NativeCliRunner
from .filesystem_jail import BubblewrapJail

__all__ = ["BubblewrapJail", "NativeCliInvocation", "NativeCliRunner"]
