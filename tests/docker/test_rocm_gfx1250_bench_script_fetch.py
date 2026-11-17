# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Supply-chain tests for the gfx1250 bench scripts baked into the image.

``docker/Dockerfile.rocm_gfx1250`` downloads ``vllm_smoketest.sh`` and
``vllm_benchserve.sh`` in the ``final`` stage. ``vllm-openai`` is
``FROM final`` and does not delete them. The downloads use ``curl -k``
against a ``refs/heads/`` URL, so certificate checks are off and the
branch tip is not pinned.

These tests replay every in-scope ``RUN`` under a local HTTPS server that
answers for ``raw.githubusercontent.com`` with an attacker-controlled
shell script. They fail while that script is what ends up in the
``vllm-openai`` image and runs as the image's default user. They pass
when those downloads are gone, or when a pinned fetch keeps only bytes
that match a checksum recorded in the Dockerfile (a substituted body
must not be left in the image).

Other fetches in this Dockerfile are not the same bug and are not
replayed: the uv installer (``curl -LsSf`` of ``uv/install.sh``), the
versioned sccache tarball, the apt signing-key downloads, and the
triton ``git clone`` / ``git checkout``. None of those disable TLS
verification or leave a branch-tip shell script in the serving image.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE_PATH = REPO_ROOT / "docker" / "Dockerfile.rocm_gfx1250"
HOST = "raw.githubusercontent.com"
MARKER = "GFX1250_BENCH_SCRIPT_MARKER"
ATTACK_SCRIPT = f"#!/bin/sh\necho {MARKER}\nid -u\n".encode()
SCRIPT_NAMES = ("vllm_smoketest.sh", "vllm_benchserve.sh")
SMOKE_URL = (
    "https://raw.githubusercontent.com/ROCm/vllm/refs/heads/"
    "gfx1250_bench/vllm_smoketest.sh"
)
BASE_IMAGE = "vllm-gfx1250-fetch-test-base:24.04"
SERVER_NAME = "gfx1250-bench-mitm"
TARGET_STAGE = "vllm-openai"

_INFRA_FAILURES = (
    "Couldn't connect",
    "Could not resolve",
    "Temporary failure in name resolution",
    "Connection refused",
    "Failed to fetch",
    "Unable to locate package",
)


@dataclass
class Stage:
    name: str
    parent: str | None
    instructions: list[tuple[str, str]] = field(default_factory=list)
    workdir: str = "/"
    user: str | None = None


@dataclass
class ParsedDockerfile:
    stages: dict[str, Stage]
    build_args: dict[str, str]


def _logical_lines(text: str) -> list[str]:
    logical: list[str] = []
    buffer = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not buffer and (not stripped or stripped.startswith("#")):
            continue
        if raw.rstrip().endswith("\\"):
            buffer += raw.rstrip()[:-1] + " "
            continue
        buffer += raw.strip() if not buffer else raw.strip()
        # Continued lines were accumulated without their newline.
        if buffer and not raw.rstrip().endswith("\\"):
            logical.append(buffer.strip())
            buffer = ""
    if buffer.strip():
        logical.append(buffer.strip())
    return logical


def _expand(value: str, variables: dict[str, str]) -> str:
    def braced(match: re.Match[str]) -> str:
        body = match.group(1)
        if ":-" in body:
            name, default = body.split(":-", 1)
            current = variables.get(name)
            return default if current is None or current == "" else current
        return variables.get(body, match.group(0))

    value = re.sub(r"\$\{([^}]+)\}", braced, value)
    return re.sub(
        r"\$([A-Za-z_][A-Za-z0-9_]*)",
        lambda match: variables.get(match.group(1), match.group(0)),
        value,
    )


def _split_instruction(line: str) -> tuple[str, str]:
    match = re.match(r"^([A-Za-z]+)\s*(.*)$", line, re.DOTALL)
    if match is None:
        return "", line
    return match.group(1).upper(), match.group(2).strip()


def _parse_arg(args: str) -> tuple[str, str | None]:
    if "=" not in args:
        return args.strip(), None
    name, default = args.split("=", 1)
    return name.strip(), default.strip().strip("\"'")


def _parse_from(args: str) -> tuple[str, str | None]:
    tokens = [token for token in args.split() if not token.startswith("--")]
    image = tokens[0]
    name = tokens[2] if len(tokens) >= 3 and tokens[1].upper() == "AS" else None
    return image, name


def parse_dockerfile(text: str) -> ParsedDockerfile:
    """Parse the stage graph, build-args, WORKDIR, and USER.

    ``ARG`` declared before the first ``FROM`` is a build-arg. A later
    ``ARG`` with the same name and no default receives that value, which
    is how ``COMMON_WORKDIR`` becomes ``/app`` for ``WORKDIR``.
    """
    stages: dict[str, Stage] = {}
    build_args: dict[str, str] = {}
    current: Stage | None = None
    for line in _logical_lines(text):
        op, args = _split_instruction(line)
        if op == "FROM":
            image, name = _parse_from(args)
            image = _expand(image, build_args)
            parent = stages.get(image)
            stage_name = name or image
            current = Stage(
                name=stage_name,
                parent=parent.name if parent else None,
                workdir=parent.workdir if parent else "/",
                user=parent.user if parent else None,
            )
            stages[stage_name] = current
            continue
        if current is None:
            if op == "ARG":
                name, default = _parse_arg(args)
                build_args[name] = "" if default is None else default
            continue
        current.instructions.append((op, args))
        if op == "ARG":
            name, default = _parse_arg(args)
            if default is None:
                build_args[name] = build_args.get(name, "")
            elif name not in build_args:
                build_args[name] = default
        elif op == "WORKDIR":
            current.workdir = _expand(args, build_args) or "/"
        elif op == "USER":
            current.user = _expand(args, build_args)
    return ParsedDockerfile(stages=stages, build_args=build_args)


def lineage_of(parsed: ParsedDockerfile, target: str) -> list[Stage]:
    chain: list[Stage] = []
    seen: set[str] = set()
    current: str | None = target
    while current is not None and current not in seen:
        seen.add(current)
        stage = parsed.stages[current]
        chain.append(stage)
        current = stage.parent
    chain.reverse()
    return chain


def _invocations(command: str) -> list[list[str]]:
    invocations: list[list[str]] = []
    for part in re.split(r"&&|\|\||;", command):
        part = part.strip()
        if not part:
            continue
        try:
            invocations.append(shlex.split(part))
        except ValueError:
            invocations.append(part.split())
    return invocations


def _disables_tls(argv: list[str]) -> bool:
    if not argv:
        return False
    program = os.path.basename(argv[0])
    if program not in {"curl", "wget"}:
        return False
    for arg in argv[1:]:
        if arg in {"-k", "--insecure", "--no-check-certificate"}:
            return True
        if (
            program == "curl"
            and arg.startswith("-")
            and not arg.startswith("--")
            and "k" in arg[1:]
        ):
            return True
    return False


def run_in_scope(command: str) -> bool:
    """True for the insecure or unpinned bench-script downloads.

    A later ``RUN`` that only checksums or deletes those scripts is in
    scope too, so a fetch that happens again after a check is still
    replayed.
    """
    if "refs/heads/" in command or "--insecure" in command:
        return True
    if any(name in command for name in SCRIPT_NAMES):
        return True
    return any(_disables_tls(argv) for argv in _invocations(command))


def selected_runs(text: str, target: str = TARGET_STAGE) -> list[str]:
    parsed = parse_dockerfile(text)
    commands: list[str] = []
    for stage in lineage_of(parsed, target):
        for op, args in stage.instructions:
            if op == "RUN" and run_in_scope(args):
                commands.append(args)
    return commands


def stage_context(text: str, target: str = TARGET_STAGE) -> tuple[str, str | None]:
    parsed = parse_dockerfile(text)
    stages = lineage_of(parsed, target)
    workdir = "/"
    user: str | None = None
    for stage in stages:
        for op, args in stage.instructions:
            if op == "RUN" and run_in_scope(args):
                return stage.workdir, stage.user
            if op == "WORKDIR":
                workdir = stage.workdir
            elif op == "USER":
                user = stage.user
        workdir = stage.workdir
        user = stage.user
    return workdir, user


def render_replay(text: str, *, trust_ca: bool) -> str:
    """Rebuild the tail that ships the bench scripts.

    Ancestor stages are flattened into ``final`` because ``FROM`` keeps
    their filesystem, then ``vllm-openai`` is ``FROM final`` with the
    child instructions from the real Dockerfile. Harness-only steps are
    the base image (distro ``curl``) and, for the branch-substitution
    case, a CA the client trusts.
    """
    parsed = parse_dockerfile(text)
    runs = selected_runs(text)
    workdir, user = stage_context(text, TARGET_STAGE)
    child = parsed.stages[TARGET_STAGE]
    child_lines = [
        f"{op} {args}".rstrip()
        for op, args in child.instructions
        if not (op == "RUN" and run_in_scope(args))
    ]
    lines = [
        f"FROM {BASE_IMAGE} AS final",
        f"WORKDIR {workdir}",
    ]
    if user:
        lines.append(f"USER {user}")
    if trust_ca:
        lines.extend(
            [
                "COPY ca.crt /usr/local/share/ca-certificates/gfx1250-test.crt",
                "RUN update-ca-certificates",
            ]
        )
    lines.extend(f"RUN {command}" for command in runs)
    final = parsed.stages.get("final")
    if final is not None:
        for op, args in final.instructions:
            if op in {"CMD", "ENTRYPOINT", "USER"}:
                lines.append(f"{op} {args}".rstrip())
    lines.append(f"FROM final AS {TARGET_STAGE}")
    lines.extend(child_lines)
    return "\n".join(lines) + "\n"


def _docker() -> list[str]:
    probe = subprocess.run(
        ["docker", "info"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if probe.returncode == 0:
        return ["docker"]
    return ["sudo", "-n", "docker"]


def _run(
    args: list[str],
    *,
    check: bool = False,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        args,
        input=input_bytes,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {args}\n"
            f"{completed.stdout}\n{completed.stderr}"
        )
    return completed


def _ensure_base_image() -> None:
    docker = _docker()
    inspect = _run(docker + ["image", "inspect", BASE_IMAGE])
    if inspect.returncode == 0:
        return
    dockerfile = textwrap.dedent(
        """\
        FROM ubuntu:24.04
        ENV DEBIAN_FRONTEND=noninteractive
        RUN apt-get update -qq \\
            && apt-get install -y -qq curl ca-certificates python3 \\
            && curl --version
        """
    )
    context = Path("/tmp/gfx1250-base-image")
    context.mkdir(parents=True, exist_ok=True)
    (context / "Dockerfile").write_text(dockerfile)
    _run(
        docker
        + [
            "build",
            "--network=host",
            "-t",
            BASE_IMAGE,
            str(context),
        ],
        check=True,
    )


def _write_certs(directory: Path) -> None:
    san = directory / "san.cnf"
    san.write_text("subjectAltName=DNS:raw.githubusercontent.com\n")
    _run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(directory / "ca.key"),
            "-out",
            str(directory / "ca.crt"),
            "-days",
            "1",
            "-subj",
            "/CN=gfx1250-test-ca",
        ],
        check=True,
    )
    _run(
        [
            "openssl",
            "req",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-keyout",
            str(directory / "server.key"),
            "-out",
            str(directory / "server.csr"),
            "-subj",
            f"/CN={HOST}",
        ],
        check=True,
    )
    _run(
        [
            "openssl",
            "x509",
            "-req",
            "-in",
            str(directory / "server.csr"),
            "-CA",
            str(directory / "ca.crt"),
            "-CAkey",
            str(directory / "ca.key"),
            "-CAcreateserial",
            "-out",
            str(directory / "server.crt"),
            "-days",
            "1",
            "-extfile",
            str(san),
        ],
        check=True,
    )


def _start_server(directory: Path) -> None:
    script = textwrap.dedent(
        """\
        import ssl
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        BODY = open("/work/body.sh", "rb").read()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(BODY)))
                self.end_headers()
                self.wfile.write(BODY)
                with open("/work/hits.log", "a", encoding="utf-8") as hits:
                    hits.write(self.path + "\\n")

            def log_message(self, fmt, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 443), Handler)
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain("/work/server.crt", "/work/server.key")
        server.socket = context.wrap_socket(server.socket, server_side=True)
        print("ready", flush=True)
        server.serve_forever()
        """
    )
    (directory / "server.py").write_text(script)
    (directory / "body.sh").write_bytes(ATTACK_SCRIPT)
    (directory / "hits.log").write_text("")
    docker = _docker()
    _run(docker + ["rm", "-f", SERVER_NAME])
    started = _run(
        docker
        + [
            "run",
            "-d",
            "--name",
            SERVER_NAME,
            "--network=host",
            "-v",
            f"{directory}:/work:ro",
            "-v",
            f"{directory}/hits.log:/work/hits.log",
            "--entrypoint",
            "python3",
            BASE_IMAGE,
            "/work/server.py",
        ]
    )
    if started.returncode != 0:
        raise AssertionError(started.stderr)
    logs = ""
    for _ in range(50):
        logs = _run(docker + ["logs", SERVER_NAME]).stdout
        if "ready" in logs:
            return
        if "Error" in logs or "Traceback" in logs:
            break
        subprocess.run(["sleep", "0.1"], check=False)
    raise AssertionError(f"mitm server did not start:\n{logs}")


def _stop_server() -> None:
    _run(_docker() + ["rm", "-f", SERVER_NAME])


def _hits(directory: Path) -> str:
    return (directory / "hits.log").read_text(encoding="utf-8")


def _curl(image: str, args: list[str]) -> subprocess.CompletedProcess[str]:
    return _run(
        _docker()
        + [
            "run",
            "--rm",
            "--network=host",
            "--add-host",
            f"{HOST}:127.0.0.1",
            "--entrypoint",
            "curl",
            image,
            *args,
        ]
    )


def _docker_build(
    context: Path,
    tag: str,
    *,
    add_host: bool,
) -> subprocess.CompletedProcess[str]:
    command = _docker() + ["build", "--network=host"]
    if add_host:
        command += ["--add-host", f"{HOST}:127.0.0.1"]
    command += ["-t", tag, str(context)]
    return _run(command)


def _classify_build(completed: subprocess.CompletedProcess[str]) -> str:
    log = completed.stdout + "\n" + completed.stderr
    if completed.returncode == 0:
        return "built"
    infra = any(marker in log for marker in _INFRA_FAILURES)
    tls_rejected = "SSL certificate problem" in log
    if infra and not tls_rejected:
        return "infra"
    return "rejected"


def _image_execution(tag: str) -> subprocess.CompletedProcess[str]:
    script = textwrap.dedent(
        f"""\
        echo "uid=$(id -u)"
        echo "pwd=$(pwd)"
        find / -xdev -type f -size -2000000 2>/dev/null |
        while IFS= read -r file; do
            if grep -q -F "{MARKER}" "$file" 2>/dev/null; then
                echo "FOUND $file"
                bash "$file"
            fi
        done
        """
    )
    return _run(
        _docker()
        + [
            "run",
            "--rm",
            "--entrypoint",
            "/bin/bash",
            tag,
            "-lc",
            script,
        ]
    )


def _apply_runs(
    tmp_path: Path,
    *,
    source: str,
    base: str,
    runs: list[str],
    workdir: str,
    user: str | None,
    tag: str,
) -> str:
    """Commit each product ``RUN`` that succeeds.

    A rejected ``RUN`` does not stop the replay. The next instruction is
    applied to the last committed image, so a download that follows a
    checksum is still executed when the checksum does not match the
    attacker body.
    """
    current = base
    logs: list[str] = []
    for index, command in enumerate(runs):
        step = tmp_path / f"step-{index}"
        step.mkdir()
        lines = [f"FROM {current}", f"WORKDIR {workdir}"]
        if user:
            lines.append(f"USER {user}")
        lines.append(f"RUN {command}")
        (step / "Dockerfile").write_text("\n".join(lines) + "\n")
        step_tag = f"{tag}-step{index}"
        built = _docker_build(step, step_tag, add_host=True)
        log = built.stdout + "\n" + built.stderr
        logs.append(log)
        kind = _classify_build(built)
        if kind == "infra":
            raise AssertionError(f"RUN could not be judged:\n{command}\n{log}")
        if kind == "built":
            current = step_tag
    parsed = parse_dockerfile(source)
    child_lines = [
        f"{op} {args}".rstrip()
        for op, args in parsed.stages[TARGET_STAGE].instructions
        if not (op == "RUN" and run_in_scope(args))
    ]
    final_dir = tmp_path / "vllm-openai-stage"
    final_dir.mkdir()
    final_text = "\n".join([f"FROM {current} AS {TARGET_STAGE}", *child_lines])
    (final_dir / "Dockerfile").write_text(final_text + "\n")
    final = _docker_build(final_dir, tag, add_host=False)
    if final.returncode != 0:
        raise AssertionError(final.stdout + "\n" + final.stderr)
    combined = "\n".join(logs)
    executed = _image_execution(tag)
    observed = executed.stdout + "\n" + executed.stderr
    assert MARKER not in observed, (
        "attacker bench script was stored in the vllm-openai image and "
        "ran as the image default user\n" + observed
    )
    exercised = any(name in _hits(tmp_path) for name in SCRIPT_NAMES) or (
        "SSL certificate problem" in combined or "sha256sum" in combined
    )
    if not exercised:
        raise AssertionError(
            "the bench-script fetch was not exercised against the test server\n"
            + combined
        )
    return tag


def _exercise(tmp_path: Path, *, trust_ca: bool, tag: str) -> None:
    source = DOCKERFILE_PATH.read_text(encoding="utf-8")
    runs = selected_runs(source)
    if not runs:
        return
    _ensure_base_image()
    _write_certs(tmp_path)
    try:
        _start_server(tmp_path)
        if trust_ca:
            trusted = tmp_path / "trusted"
            trusted.mkdir()
            shutil.copy(tmp_path / "ca.crt", trusted / "ca.crt")
            (trusted / "Dockerfile").write_text(
                "FROM "
                + BASE_IMAGE
                + "\nCOPY ca.crt /usr/local/share/ca-certificates/gfx1250-test.crt\n"
                "RUN update-ca-certificates\n"
            )
            trusted_image = tag + "-trusted-base"
            _run(
                _docker()
                + ["build", "--network=host", "-t", trusted_image, str(trusted)],
                check=True,
            )
            preflight = _curl(trusted_image, ["-fsS", SMOKE_URL])
            if preflight.returncode != 0 or MARKER not in preflight.stdout:
                raise AssertionError(
                    "trusted-cert preflight did not receive the attacker body\n"
                    + preflight.stdout
                    + preflight.stderr
                )
            (tmp_path / "hits.log").write_text("")
            base_for_replay = trusted_image
        else:
            rejected = _curl(BASE_IMAGE, ["-fsS", SMOKE_URL])
            if rejected.returncode == 0:
                raise AssertionError(
                    "curl without -k accepted the untrusted certificate"
                )
            accepted = _curl(BASE_IMAGE, ["-k", "-fsS", SMOKE_URL])
            if accepted.returncode != 0 or MARKER not in accepted.stdout:
                raise AssertionError(
                    "curl -k did not accept the untrusted certificate\n"
                    + accepted.stderr
                )
            (tmp_path / "hits.log").write_text("")
            base_for_replay = BASE_IMAGE
        workdir, user = stage_context(source)
        _apply_runs(
            tmp_path,
            source=source,
            base=base_for_replay,
            runs=runs,
            workdir=workdir,
            user=user,
            tag=tag,
        )
    finally:
        _stop_server()


def test_selector_keeps_a_download_that_follows_a_checksum() -> None:
    """A checksum does not end the replay.

    The fetch that runs after the check still has to be executed, or a
    fix could validate one path and publish a second download.
    """
    dockerfile = textwrap.dedent(
        """\
        FROM scratch AS final
        RUN curl -fsSL -o vllm_smoketest.sh https://example.invalid/pinned \\
            && echo abcdef vllm_smoketest.sh | sha256sum -c -
        RUN curl -k -O https://example.invalid/refs/heads/x/vllm_smoketest.sh
        FROM final AS vllm-openai
        ENTRYPOINT ["vllm", "serve"]
        """
    )
    runs = selected_runs(dockerfile)
    assert len(runs) == 2
    assert "sha256sum" in runs[0]
    assert "curl -k -O" in runs[1]
    assert runs[1] in render_replay(dockerfile, trust_ca=False)


def test_parser_resolves_unpinned_downloads_in_the_serving_stage() -> None:
    source = DOCKERFILE_PATH.read_text(encoding="utf-8")
    parsed = parse_dockerfile(source)
    names = [stage.name for stage in lineage_of(parsed, TARGET_STAGE)]
    assert names[0] == "base"
    assert names[-1] == TARGET_STAGE
    assert "final" in names
    runs = selected_runs(source)
    # Removal of the downloads passes the planting tests. While they
    # remain, the selector must see both of the serving-stage fetches.
    if runs:
        assert any("vllm_smoketest.sh" in command for command in runs)
        assert any("vllm_benchserve.sh" in command for command in runs)
        workdir, user = stage_context(source)
        assert workdir == "/app"
        assert user is None


def test_untrusted_tls_cannot_supply_gfx1250_bench_scripts(tmp_path: Path) -> None:
    """A certificate curl would reject must not become a bench script.

    The server certificate is self-signed. ``curl`` without ``-k`` refuses
    it. The image ``RUN`` lines are then executed unchanged. The planted
    script must not be present in the ``vllm-openai`` stage.
    """
    _exercise(tmp_path, trust_ca=False, tag="gfx1250-bench-untrusted")


def test_branch_tip_substitution_cannot_supply_gfx1250_bench_scripts(
    tmp_path: Path,
) -> None:
    """Bytes from the branch tip must not be kept unless they were pinned.

    The same server certificate is signed by a CA the client trusts, which
    is the branch-push case: TLS succeeds and the body is whatever the
    branch currently serves. The image must not keep that body as
    ``vllm_smoketest.sh`` or ``vllm_benchserve.sh``.
    """
    _exercise(tmp_path, trust_ca=True, tag="gfx1250-bench-branch")
