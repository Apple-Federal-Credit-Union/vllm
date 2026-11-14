# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
import time

import jinja2
import pytest

from vllm.renderers.template_deadline import (
    ChatTemplateDeadlineError,
    render_with_deadline,
)


def _render(template: str, **context: object) -> str:
    env = jinja2.Environment()
    return render_with_deadline(lambda: env.from_string(template).render(**context))


def test_c_level_filters_stop_at_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT", "0.25")
    templates = (
        "{{ range(10 ** 15)|sum }}",
        "{{ range(10 ** 15)|list }}",
        "{{ range(10 ** 15)|join }}",
    )
    for template in templates:
        holder: dict[str, BaseException] = {}

        def run(template: str = template) -> None:
            try:
                _render(template)
            except BaseException as exc:
                holder["exc"] = exc

        started = time.monotonic()
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(5)
        assert not worker.is_alive()
        assert time.monotonic() - started < 5
        exc = holder["exc"]
        assert isinstance(exc, ChatTemplateDeadlineError)
        assert "timed out" in str(exc)


def test_small_consuming_filters_keep_their_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT", "2")
    assert (
        _render(
            "{{ range(5)|sum }}|{{ range(5)|join(',') }}"
            "|{{ range(5)|list|join('-') }}",
        )
        == "10|0,1,2,3,4|0-1-2-3-4"
    )
    assert (
        _render(
            "{{ items|sum(attribute='n') }}|{{ items|join(', ', attribute='name') }}",
            items=[{"n": 1, "name": "a"}, {"n": 2, "name": "b"}],
        )
        == "3|a, b"
    )


def test_unbounded_power_still_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT", "2")
    with pytest.raises(ChatTemplateDeadlineError, match=r"\*\*"):
        _render("{{ 2 ** n }}", n=30)
