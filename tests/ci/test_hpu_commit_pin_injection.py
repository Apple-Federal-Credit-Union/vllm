# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The HPU CI commit pin must be data, not a shell command.

``.buildkite/scripts/hardware_ci/run-hpu-test.sh`` downloads the first line of
``VLLM_COMMUNITY_COMMIT`` and pastes it into a Dockerfile ``RUN`` instruction.
These tests drive that script. ``curl`` is the download, and ``docker build``
is the real Docker client. A pin is unsafe when the image build evaluates
shell syntax that was not part of the script.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HPU_SCRIPT = REPO_ROOT / ".buildkite" / "scripts" / "hardware_ci" / "run-hpu-test.sh"

# Arithmetic is expanded only when a shell executes the pin. The expected
# strings do not appear in the pin text.
UNSAFE_PINS = {
    "semicolon": (
        "notasha; echo HPU_PIN_RAN_$((90000+7531))\n",
        "HPU_PIN_RAN_97531",
    ),
    "latest_prefix": (
        "latest; echo HPU_PIN_RAN_$((96000+7531))\n",
        "HPU_PIN_RAN_103531",
    ),
    "command_substitution": (
        "$(echo HPU_PIN_RAN_$((93000+7531)) >&2)\n",
        "HPU_PIN_RAN_100531",
    ),
    "backticks": (
        "`echo HPU_PIN_RAN_$((94000+7531)) >&2`\n",
        "HPU_PIN_RAN_101531",
    ),
    "unanchored_sha_prefix": (
        "0123456789abcdef0123456789abcdef01234567; "
        "echo HPU_PIN_RAN_$((95000+7531))\n",
        "HPU_PIN_RAN_102531",
    ),
    "quote_breakout": (
        '"; echo HPU_PIN_RAN_$((92000+7531)); "\n',
        "HPU_PIN_RAN_99531",
    ),
}


def _ensure_base_image() -> None:
    inspect = subprocess.run(
        ["docker", "image", "inspect", "gaudi-base-image:latest"],
        capture_output=True,
        check=False,
    )
    if inspect.returncode == 0:
        return
    subprocess.run(["docker", "pull", "alpine:3.20"], check=True)
    subprocess.run(
        ["docker", "tag", "alpine:3.20", "gaudi-base-image:latest"],
        check=True,
    )


def _run_script(tmp_path: Path, bodies: list[str], commit: str) -> str:
    _ensure_base_image()
    bin_dir = tmp_path / "bin"
    context = tmp_path / "context"
    body_dir = tmp_path / "bodies"
    bin_dir.mkdir()
    context.mkdir()
    body_dir.mkdir()
    # A unique context file changes the COPY layer so Docker cannot reuse a
    # cached RUN and hide a command that the pin still executes.
    (context / "README").write_text(commit + "\n", encoding="utf-8")
    for index, body in enumerate(bodies):
        (body_dir / str(index)).write_text(body, encoding="utf-8")
    index_file = tmp_path / "curl-index"
    index_file.write_text("0", encoding="utf-8")
    curl = bin_dir / "curl"
    curl.write_text(
        "\n".join(
            [
                "#!/bin/sh",
                f'idx=$(cat "{index_file}")',
                f'next=$((idx + 1))',
                f'printf "%s" "$next" > "{index_file}"',
                f'cat "{body_dir}/$idx"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    curl.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env.get('PATH', '')}"
    env["BUILDKITE_COMMIT"] = commit
    env["BUILDKIT_PROGRESS"] = "plain"
    env.pop("HPU_PIN_PATH", None)
    completed = subprocess.run(
        ["bash", str(HPU_SCRIPT)],
        cwd=context,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    output = completed.stdout + completed.stderr
    image = f"hpu/upstream-vllm-ci:{commit}"
    subprocess.run(["docker", "rmi", "-f", image], capture_output=True, check=False)
    return output


@pytest.mark.parametrize("case", sorted(UNSAFE_PINS))
def test_downloaded_pin_is_not_executed(tmp_path: Path, case: str) -> None:
    body, executed = UNSAFE_PINS[case]
    output = _run_script(tmp_path, [body], f"pin-{case}")
    assert executed not in output


def test_exact_latest_is_data(tmp_path: Path) -> None:
    output = _run_script(tmp_path, ["latest\n"], "pin-latest")
    assert "HPU_PIN_RAN_" not in output
    assert "Step 1/" in output or "FROM gaudi-base-image:latest" in output


def test_exact_sha_is_data(tmp_path: Path) -> None:
    sha = "0123456789abcdef0123456789abcdef01234567\n"
    output = _run_script(tmp_path, [sha], "pin-sha")
    assert "HPU_PIN_RAN_" not in output
    assert "Step 1/" in output or "FROM gaudi-base-image:latest" in output


def test_pin_is_not_reread_after_it_is_accepted(tmp_path: Path) -> None:
    """A later download must not replace a pin that was already accepted."""
    malicious, executed = UNSAFE_PINS["semicolon"]
    output = _run_script(tmp_path, ["latest\n", malicious], "pin-reread")
    assert executed not in output
    assert "Step 1/" in output or "FROM gaudi-base-image:latest" in output


def test_docker_client_is_real() -> None:
    docker = shutil.which("docker")
    assert docker is not None
    completed = subprocess.run(
        [docker, "version", "--format", "{{.Client.Version}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip()
