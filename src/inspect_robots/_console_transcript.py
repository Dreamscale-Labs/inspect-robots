"""Echo each policy turn into the terminal while the rollout runs.

``run --show-transcript`` attaches :class:`ConsoleTranscriptSink`. It receives
the live transcript delta through the duck-typed ``log_policy_messages`` sink
extension and writes assistant turns (recorded reasoning, reply text, and tool
calls) as scrollback. On attended runs the writer is
``OperatorSession.write_line``, which keeps the operator console's status and
input rows intact; a policy printing to stderr directly would be overdrawn by
the next status refresh.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Sequence
from typing import Any

from inspect_robots.logging.sink import NullSink

Styler = Callable[[str, str], str]

_DIM = "2"
_CYAN = "36"
_YELLOW = "33"


def _plain(text: str, code: str) -> str:
    """Return ``text`` unstyled; the default when no terminal styler is supplied."""
    del code
    return text


class ConsoleTranscriptSink(NullSink):
    """Write each assistant turn's reasoning, reply, and tool calls as scrollback.

    ``write_line`` receives one line at a time and must not break an open
    console footer (``OperatorSession.write_line`` does not; plain ``print``
    suits unattended runs). The first rendering or write failure disables the
    sink with one stderr notice instead of reaching the rollout: echo is a
    convenience and must never stop a robot run.
    """

    def __init__(self, write_line: Callable[[str], None], *, style: Styler = _plain) -> None:
        self._write_line = write_line
        self._style = style
        self._disabled = False

    def log_policy_messages(self, t: int, messages: Sequence[Any]) -> None:
        """Render the assistant turns in one live transcript delta."""
        if self._disabled:
            return
        try:
            for message in messages:
                for line in _assistant_lines(t, message, self._style):
                    self._write_line(line)
        except Exception as exc:
            self._disabled = True
            print(f"transcript echo disabled: {exc}", file=sys.stderr)


def _assistant_lines(t: int, message: object, style: Styler) -> list[str]:
    """Format one assistant message, or nothing for any other role or shape."""
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return []
    lines: list[str] = []
    reasoning = message.get("reasoning")
    if isinstance(reasoning, str) and reasoning.strip():
        lines.extend(style(f"  │ {line}", _DIM) for line in reasoning.strip().splitlines())
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        lines.extend(style(f"  » {line}", _CYAN) for line in content.strip().splitlines())
    tool_calls = message.get("tool_calls")
    for call in tool_calls if isinstance(tool_calls, list) else []:
        function = call.get("function") if isinstance(call, dict) else None
        if not isinstance(function, dict):
            continue
        arguments = function.get("arguments", "")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        lines.append(style(f"  → {function.get('name', 'unknown')}({arguments})", _YELLOW))
    if not lines:
        return []
    return [style(f"── agent turn at step {t}", _DIM), *lines]
