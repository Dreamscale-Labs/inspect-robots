"""The ``--show-transcript`` console sink renders assistant turns and never breaks a run."""

from __future__ import annotations

import argparse
from typing import cast

import pytest

from inspect_robots import cli
from inspect_robots._console_transcript import ConsoleTranscriptSink
from inspect_robots.session import OperatorSession


def _sink() -> tuple[ConsoleTranscriptSink, list[str]]:
    lines: list[str] = []
    return ConsoleTranscriptSink(lines.append), lines


def test_assistant_turn_renders_reasoning_reply_and_tool_calls_in_order() -> None:
    sink, lines = _sink()

    sink.log_policy_messages(
        12,
        [
            {"role": "user", "content": "observation"},
            {
                "role": "assistant",
                "reasoning": "cube is left\nmove left first",
                "content": "Moving left.",
                "tool_calls": [
                    {"function": {"name": "move_joints", "arguments": '{"left_j0": 0.1}'}},
                    {"function": {"name": "done", "arguments": {"summary": "ok"}}},
                    {"function": "malformed"},
                    "malformed",
                ],
            },
            {"role": "tool", "content": "executing"},
        ],
    )

    assert lines == [
        "── agent turn at step 12",
        "  │ cube is left",
        "  │ move left first",
        "  » Moving left.",
        '  → move_joints({"left_j0": 0.1})',
        '  → done({"summary": "ok"})',
    ]


@pytest.mark.parametrize(
    "message",
    [
        "not a dict",
        {"role": "assistant", "content": "  ", "reasoning": 3, "tool_calls": "nope"},
        {"role": "assistant", "content": None},
    ],
)
def test_empty_or_malformed_assistant_messages_render_nothing(message: object) -> None:
    sink, lines = _sink()

    sink.log_policy_messages(0, [message])

    assert lines == []


def test_style_wraps_every_line() -> None:
    lines: list[str] = []
    sink = ConsoleTranscriptSink(lines.append, style=lambda text, code: f"<{code}>{text}")

    sink.log_policy_messages(1, [{"role": "assistant", "content": "hi", "reasoning": "why"}])

    assert lines == ["<2>── agent turn at step 1", "<2>  │ why", "<36>  » hi"]


def test_write_failure_disables_sink_once_without_raising(
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []

    def failing(line: str) -> None:
        calls.append(line)
        raise OSError("terminal gone")

    sink = ConsoleTranscriptSink(failing)
    turn = {"role": "assistant", "content": "hi"}

    sink.log_policy_messages(0, [turn])
    sink.log_policy_messages(1, [turn])

    assert len(calls) == 1
    assert capsys.readouterr().err == "transcript echo disabled: terminal gone\n"


def test_factory_is_off_by_default_and_prefers_the_operator_console() -> None:
    written: list[str] = []

    class _Session:
        def write_line(self, text: str) -> None:
            written.append(text)

    session = cast(OperatorSession, _Session())
    assert cli._console_transcript_sink(argparse.Namespace(show_transcript=False), session) is None

    sink = cli._console_transcript_sink(argparse.Namespace(show_transcript=True), session)
    assert sink is not None
    sink.log_policy_messages(0, [{"role": "assistant", "content": "hi"}])
    assert written[-1].endswith("» hi")


def test_factory_prints_when_unattended(capsys: pytest.CaptureFixture[str]) -> None:
    sink = cli._console_transcript_sink(argparse.Namespace(show_transcript=True), None)
    assert sink is not None

    sink.log_policy_messages(3, [{"role": "assistant", "content": "hi"}])

    assert capsys.readouterr().out == "── agent turn at step 3\n  » hi\n"


def test_plain_style_ignores_code() -> None:
    from inspect_robots._console_transcript import _plain

    assert _plain("text", "36") == "text"
