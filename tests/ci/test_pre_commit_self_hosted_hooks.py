# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pull-request hooks must not run on the self-hosted pre-commit runner.

The workflow under test checks the author and labels, checks out the pull
request merge commit, and runs pre-commit. These tests follow that sequence
and fail while a hook from the pull request still runs on a self-hosted
runner.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import textwrap

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pre-commit.yml"
PRE_COMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
PRE_COMMIT_ACTION = (
    "2c7b3805fd2a0fd8c1884dcaebf91fc102a13ecd"  # pre-commit/action v3.0.1
)
CHECKOUT_REF_RE = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
PR_REF_PATTERN = re.compile(r"^refs/pull/[0-9]+/(?:head|merge)$")
STATUS_FUNCTIONS = ("always", "success", "failure", "cancelled")

_BOOL_TAG = "tag:yaml.org,2002:bool"


class _WorkflowLoader(yaml.SafeLoader):
    """Load GitHub workflow YAML without treating the `on` key as a boolean."""


_WorkflowLoader.yaml_implicit_resolvers = {
    key: (
        [(tag, pattern) for tag, pattern in patterns if tag != _BOOL_TAG]
        if key in "oO"
        else patterns
    )
    for key, patterns in yaml.SafeLoader.yaml_implicit_resolvers.items()
}


@dataclass(frozen=True)
class Scenario:
    name: str
    merged_count: int
    labels: tuple[str, ...]
    action: str = "opened"
    event_name: str = "pull_request"
    owner: str = "vllm-project"
    hook: str = "local-manual"


@dataclass
class Replay:
    marker: Path
    ran_on_self_hosted: bool
    output: str
    checkout_sha: str | None

    def message(self) -> str:
        return (
            "pull-request hook command ran on a self-hosted runner\n"
            f"checkout={self.checkout_sha}\n"
            f"output:\n{self.output}"
        )


def _load_workflow() -> dict:
    return yaml.load(WORKFLOW_PATH.read_text(), Loader=_WorkflowLoader)


def _repo_default_stages() -> list[str]:
    loaded = yaml.safe_load(PRE_COMMIT_CONFIG.read_text())
    stages = loaded.get("default_stages")
    assert isinstance(stages, list)
    return [str(stage) for stage in stages]


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "Hook Test",
            "GIT_AUTHOR_EMAIL": "hook-test@example.com",
            "GIT_COMMITTER_NAME": "Hook Test",
            "GIT_COMMITTER_EMAIL": "hook-test@example.com",
        }
    )
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        capture_output=True,
        text=True,
        env=env,
    )


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _write_marker_script(path: Path, marker: Path) -> None:
    path.write_text(f"#!/bin/sh\necho ran > {marker}\n")
    path.chmod(0o755)


def _local_hook_config(
    script: Path,
    *,
    stages: list[str] | None,
    default_stages: list[str] | None,
) -> str:
    lines = []
    if default_stages is not None:
        lines.append("default_stages:")
        lines.extend(f"  - {stage}" for stage in default_stages)
    lines.extend(
        [
            "repos:",
            "- repo: local",
            "  hooks:",
            "  - id: marker",
            "    name: marker",
            "    entry: " + json.dumps(str(script)),
            "    language: system",
            "    always_run: true",
            "    pass_filenames: false",
        ]
    )
    if stages is not None:
        lines.append("    stages: [" + ", ".join(stages) + "]")
    return "\n".join(lines) + "\n"


def _init_consumer_repo(root: Path) -> None:
    root.mkdir(parents=True)
    _git(root, "init", "-b", "main")
    (root / "README").write_text("base\n")
    _commit_all(root, "base")


def _build_local_pr(root: Path, marker: Path, scenario: Scenario) -> tuple[str, str]:
    script = root.parent / "write_marker.sh"
    _write_marker_script(script, marker)
    if scenario.hook == "local-manual":
        config = _local_hook_config(script, stages=["manual"], default_stages=None)
    elif scenario.hook == "local-builtin-stages":
        config = _local_hook_config(script, stages=None, default_stages=None)
    elif scenario.hook == "local-repo-default-stages":
        config = _local_hook_config(
            script,
            stages=None,
            default_stages=_repo_default_stages(),
        )
    else:
        raise AssertionError(scenario.hook)
    _git(root, "checkout", "-b", "pr")
    (root / ".pre-commit-config.yaml").write_text(config)
    head = _commit_all(root, "pr hook")
    _git(root, "checkout", "main")
    _git(root, "merge", "--no-ff", "--no-edit", "pr")
    merge = _git(root, "rev-parse", "HEAD").stdout.strip()
    base = _git(root, "rev-parse", "main^1").stdout.strip()
    return head, base if base else merge


def _build_remote_pr(root: Path, marker: Path) -> tuple[str, str]:
    hook_repo = root.parent / "hook-repo"
    hook_repo.mkdir()
    _git(hook_repo, "init", "-b", "main")
    benign = hook_repo / "write_benign.sh"
    benign.write_text("#!/bin/sh\nexit 0\n")
    benign.chmod(0o755)
    (hook_repo / ".pre-commit-hooks.yaml").write_text(
        textwrap.dedent(
            f"""\
            - id: marker
              name: marker
              entry: {json.dumps(str(benign))}
              language: system
              always_run: true
              pass_filenames: false
            """
        )
    )
    benign_rev = _commit_all(hook_repo, "benign hook")
    malicious = hook_repo / "write_marker.sh"
    _write_marker_script(malicious, marker)
    manifest = hook_repo / ".pre-commit-hooks.yaml"
    manifest.write_text(
        textwrap.dedent(
            f"""\
            - id: marker
              name: marker
              entry: {json.dumps(str(malicious))}
              language: system
              always_run: true
              pass_filenames: false
            """
        )
    )
    malicious_rev = _commit_all(hook_repo, "malicious hook")

    def config(rev: str) -> str:
        return textwrap.dedent(
            f"""\
            repos:
            - repo: {json.dumps(str(hook_repo))}
              rev: {rev}
              hooks:
              - id: marker
            """
        )

    (root / ".pre-commit-config.yaml").write_text(config(benign_rev))
    _commit_all(root, "pin benign hook")
    _git(root, "checkout", "-b", "pr")
    (root / ".pre-commit-config.yaml").write_text(config(malicious_rev))
    head = _commit_all(root, "pin malicious hook")
    _git(root, "checkout", "main")
    _git(root, "merge", "--no-ff", "--no-edit", "pr")
    base = _git(root, "rev-parse", "main^1").stdout.strip()
    return head, base


def _prepare_repo(tmp: Path, marker: Path, scenario: Scenario) -> dict[str, str]:
    source = tmp / "source"
    _init_consumer_repo(source)
    if scenario.hook == "remote-rev":
        head, base = _build_remote_pr(source, marker)
    else:
        head, base = _build_local_pr(source, marker, scenario)
    merge = _git(source, "rev-parse", "HEAD").stdout.strip()
    return {
        "source": str(source),
        "head": head,
        "base": base,
        "merge": merge,
        "default_branch": "main",
    }


class _Parser:
    def __init__(self, text: str, context: dict) -> None:
        self.text = text
        self.context = context
        self.index = 0

    def parse(self):
        value = self._or()
        self._skip()
        if self.index != len(self.text):
            raise ValueError(f"unparsed expression tail: {self.text[self.index:]}")
        return value

    def _skip(self) -> None:
        while self.index < len(self.text) and self.text[self.index].isspace():
            self.index += 1

    def _or(self):
        value = self._and()
        while True:
            self._skip()
            if not self.text.startswith("||", self.index):
                return value
            self.index += 2
            right = self._and()
            value = value or right

    def _and(self):
        value = self._cmp()
        while True:
            self._skip()
            if not self.text.startswith("&&", self.index):
                return value
            self.index += 2
            right = self._cmp()
            value = value and right

    def _cmp(self):
        value = self._unary()
        self._skip()
        for operator in ("==", "!="):
            if self.text.startswith(operator, self.index):
                self.index += 2
                right = self._unary()
                return _equals(value, right) if operator == "==" else not _equals(
                    value, right
                )
        return value

    def _unary(self):
        self._skip()
        if self.text.startswith("!", self.index):
            self.index += 1
            return not self._unary()
        return self._primary()

    def _primary(self):
        self._skip()
        if self.index >= len(self.text):
            raise ValueError("truncated expression")
        char = self.text[self.index]
        if char == "(":
            self.index += 1
            value = self._or()
            self._skip()
            if self.index >= len(self.text) or self.text[self.index] != ")":
                raise ValueError("missing )")
            self.index += 1
            return value
        if char == "'":
            return self._string()
        if char.isdigit():
            start = self.index
            while self.index < len(self.text) and self.text[self.index].isdigit():
                self.index += 1
            return int(self.text[start : self.index])
        return self._name()

    def _string(self) -> str:
        self.index += 1
        chars: list[str] = []
        while self.index < len(self.text):
            char = self.text[self.index]
            self.index += 1
            if char == "'":
                if self.index < len(self.text) and self.text[self.index] == "'":
                    chars.append("'")
                    self.index += 1
                    continue
                return "".join(chars)
            chars.append(char)
        raise ValueError("unterminated string")

    def _name(self):
        start = self.index
        while self.index < len(self.text) and (
            self.text[self.index].isalnum() or self.text[self.index] in "_-"
        ):
            self.index += 1
        name = self.text[start : self.index]
        parts = [name]
        while self.index < len(self.text) and self.text[self.index] == ".":
            self.index += 1
            part_start = self.index
            while self.index < len(self.text) and (
                self.text[self.index].isalnum() or self.text[self.index] in "_-"
            ):
                self.index += 1
            parts.append(self.text[part_start : self.index])
        self._skip()
        if self.index < len(self.text) and self.text[self.index] == "(":
            self.index += 1
            self._skip()
            if self.index >= len(self.text) or self.text[self.index] != ")":
                raise ValueError(f"only nullary calls are supported, got {name}")
            self.index += 1
            if len(parts) != 1:
                raise ValueError(name)
            return _status_call(name, self.context)
        current = self.context
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current


def _equals(left, right) -> bool:
    if isinstance(left, str) and isinstance(right, str):
        return left.casefold() == right.casefold()
    return left == right


def _status_call(name: str, context: dict):
    if name not in STATUS_FUNCTIONS:
        raise ValueError(f"unsupported function {name}")
    results = [
        job.get("result")
        for job in context.get("needs", {}).values()
        if isinstance(job, dict)
    ]
    if name == "always":
        return True
    if name == "cancelled":
        return any(result == "cancelled" for result in results)
    if name == "failure":
        return any(result == "failure" for result in results)
    return all(result == "success" for result in results) if results else True


def _contains_status_function(expression: str) -> bool:
    return any(
        re.search(rf"\b{name}\s*\(", expression) is not None for name in STATUS_FUNCTIONS
    )


def _unwrap(expression: str) -> str:
    text = expression.strip()
    if text.startswith("${{") and text.endswith("}}"):
        return text[3:-2].strip()
    return text


def _eval_expr(expression: str, context: dict):
    return _Parser(_unwrap(expression), context).parse()


def _truthy(value) -> bool:
    return value not in (False, None, 0, "")


def _condition(expression: str | None, context: dict) -> bool:
    if expression is None:
        return _truthy(_status_call("success", context))
    value = _truthy(_eval_expr(expression, context))
    if not _contains_status_function(expression):
        value = value and _truthy(_status_call("success", context))
    return value


def _triggered(workflow: dict, event_name: str, action: str, ref: str) -> bool:
    events = workflow["on"]
    if event_name == "pull_request":
        pull = events.get("pull_request")
        if pull is None:
            return False
        types = pull.get("types") if isinstance(pull, dict) else None
        return types is None or action in types
    if event_name == "push":
        push = events.get("push")
        if push is None:
            return False
        branches = push.get("branches", []) if isinstance(push, dict) else []
        return ref in {f"refs/heads/{branch}" for branch in branches}
    return False


def _gate_script(workflow: dict) -> str | None:
    job = workflow["jobs"].get("pre-run-check")
    if not isinstance(job, dict):
        return None
    for step in job.get("steps", []):
        if "github-script" in str(step.get("uses", "")):
            return step.get("with", {}).get("script")
    return None


def _run_gate_script(script: str, context: dict, pr: dict, merged_count: int) -> bool:
    """Run the workflow script the way actions/github-script calls it.

    Returns True when the step fails.
    """

    runner = textwrap.dedent(
        """\
        const fs = require('fs');
        const input = JSON.parse(fs.readFileSync(0, 'utf8'));
        const AsyncFunction = Object.getPrototypeOf(async () => null).constructor;
        const failed = [];
        const core = {
          info() {},
          setFailed(message) {
            failed.push(String(message));
            process.exitCode = 1;
          },
        };
        const github = {
          rest: {
            pulls: {get: async () => ({data: input.pr})},
            search: {
              issuesAndPullRequests: async () => ({
                data: {total_count: input.mergedCount},
              }),
            },
          },
        };
        const args = {
          require,
          __original_require__: require,
          github,
          octokit: github,
          getOctokit: () => github,
          context: input.context,
          core,
          exec: {},
          glob: {},
          io: {},
        };
        const fn = new AsyncFunction(...Object.keys(args), input.script);
        fn(...Object.values(args))
          .then(() => {
            process.stdout.write(JSON.stringify({failed}));
          })
          .catch((error) => {
            process.stdout.write(JSON.stringify({
              failed,
              error: String(error),
            }));
            process.exitCode = 1;
          });
        """
    )
    completed = subprocess.run(
        ["node", "-e", runner],
        input=json.dumps(
            {
                "script": script,
                "context": context,
                "pr": pr,
                "mergedCount": merged_count,
            }
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    if not completed.stdout:
        raise AssertionError(completed.stderr)
    payload = json.loads(completed.stdout)
    return bool(payload.get("failed") or payload.get("error"))


def _push_sha(target: list[str], value) -> None:
    if isinstance(value, str) and value:
        target.append(value.lower())


def _assert_safe_pr_checkout(
    event_name: str,
    payload: dict,
    qualified_repository: str,
    ref: str,
    commit: str | None,
    allow: bool,
) -> None:
    """Port of actions/checkout assertSafePrCheckout at the pinned commit.

    `pull_request` returns before any fork comparison.
    """

    if allow:
        return
    if event_name not in {"pull_request_target", "workflow_run"}:
        return
    base_repo_id = (payload.get("repository") or {}).get("id")
    if not isinstance(base_repo_id, int):
        return
    pr_shas: list[str] = []
    if event_name == "pull_request_target":
        pull = payload.get("pull_request") or {}
        head = pull.get("head") or {}
        head_repo = head.get("repo") or {}
        pr_head_repo_id = head_repo.get("id")
        pr_head_repo_full_name = head_repo.get("full_name")
        _push_sha(pr_shas, head.get("sha"))
        _push_sha(pr_shas, pull.get("merge_commit_sha"))
    else:
        workflow_run = payload.get("workflow_run") or {}
        wr_event = workflow_run.get("event")
        if not isinstance(wr_event, str) or not wr_event.startswith("pull_request"):
            return
        head_repo = workflow_run.get("head_repository") or {}
        pr_head_repo_id = head_repo.get("id")
        pr_head_repo_full_name = head_repo.get("full_name")
        head_commit = workflow_run.get("head_commit") or {}
        _push_sha(pr_shas, head_commit.get("id"))
        if wr_event != "pull_request_target":
            _push_sha(pr_shas, workflow_run.get("head_sha"))
    if not isinstance(pr_head_repo_id, int) or pr_head_repo_id == base_repo_id:
        return
    repository_matches = (
        isinstance(pr_head_repo_full_name, str)
        and qualified_repository.lower() == pr_head_repo_full_name.lower()
    )
    ref_matches = PR_REF_PATTERN.fullmatch(ref or "") is not None
    commit_matches = bool(commit) and commit.lower() in pr_shas
    if not (repository_matches or ref_matches or commit_matches):
        return
    raise RuntimeError("unsafe fork checkout")


def _checkout_inputs(step: dict, github: dict) -> tuple[str, str | None, bool]:
    raw = (step.get("with") or {}).get("ref")
    ref = "" if raw is None else str(_eval_expr(str(raw), {"github": github}))
    commit = None
    if not ref:
        ref = github["ref"]
        commit = github["sha"]
    elif CHECKOUT_REF_RE.fullmatch(ref):
        commit = ref
        ref = ""
    allow_raw = (step.get("with") or {}).get("allow-unsafe-pr-checkout", "false")
    allow = str(allow_raw).strip().lower() == "true"
    return ref, commit, allow


def _fetch(workspace: Path, source: str, *refspecs: str, depth: int) -> None:
    command = [
        "git",
        "-c",
        "protocol.version=2",
        "fetch",
        "--no-tags",
        "--prune",
        "--no-recurse-submodules",
        "--depth",
        str(depth),
        source,
        *refspecs,
    ]
    completed = subprocess.run(
        command,
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr)


def _checkout_tree(
    source: str,
    workspace: Path,
    ref: str,
    commit: str | None,
) -> str:
    """Fetch and check out the ref the way actions/checkout does."""

    workspace.mkdir(parents=True)
    _git(workspace, "init")
    _git(workspace, "remote", "add", "origin", source)
    upper = (ref or "").upper()
    if commit and upper.startswith("REFS/PULL/"):
        branch = ref[len("refs/pull/") :]
        _fetch(
            workspace,
            source,
            f"+{commit}:refs/remotes/pull/{branch}",
            depth=1,
        )
        _git(
            workspace,
            "checkout",
            "--force",
            f"refs/remotes/pull/{branch}",
        )
    elif commit and upper.startswith("REFS/HEADS/"):
        branch = ref[len("refs/heads/") :]
        _fetch(
            workspace,
            source,
            f"+{commit}:refs/remotes/origin/{branch}",
            depth=1,
        )
        _git(
            workspace,
            "checkout",
            "--force",
            "-B",
            branch,
            f"refs/remotes/origin/{branch}",
        )
    elif not commit and ref and not upper.startswith("REFS/"):
        _fetch(
            workspace,
            source,
            f"+refs/heads/{ref}*:refs/remotes/origin/{ref}*",
            depth=1,
        )
        _git(
            workspace,
            "checkout",
            "--force",
            "-B",
            ref,
            f"refs/remotes/origin/{ref}",
        )
    elif not commit and upper.startswith("REFS/HEADS/"):
        branch = ref[len("refs/heads/") :]
        _fetch(
            workspace,
            source,
            f"+{ref}:refs/remotes/origin/{branch}",
            depth=1,
        )
        _git(
            workspace,
            "checkout",
            "--force",
            "-B",
            branch,
            f"refs/remotes/origin/{branch}",
        )
    elif not ref and commit:
        _fetch(workspace, source, commit, depth=1)
        _git(workspace, "checkout", "--force", commit)
    else:
        raise AssertionError(f"unsupported checkout ref={ref!r} commit={commit!r}")
    return _git(workspace, "rev-parse", "HEAD").stdout.strip()


def _runs_on_labels(runs_on, context: dict) -> list[str]:
    if isinstance(runs_on, str):
        value = _eval_expr(runs_on, context) if "${{" in runs_on else runs_on
        return [str(value)]
    if isinstance(runs_on, list):
        return [str(label) for label in runs_on]
    if isinstance(runs_on, dict):
        labels = runs_on.get("labels", [])
        if isinstance(labels, str):
            return [labels]
        return [str(label) for label in labels]
    return []


def _self_hosted(labels: list[str]) -> bool:
    return any(label.casefold() == "self-hosted" for label in labels)


def _pre_commit_command(step: dict) -> str:
    extra = "--all-files"
    if step.get("with") and step["with"].get("extra_args"):
        extra = str(step["with"]["extra_args"])
    # pre-commit/action composite step interpolates extra_args unquoted.
    return f"pre-commit run --show-diff-on-failure --color=always {extra}"


def _run_steps(
    steps: list[dict],
    *,
    source: str,
    workspace: Path,
    github: dict,
    payload: dict,
    pre_commit_home: Path,
) -> tuple[str | None, str]:
    output: list[str] = []
    checkout_sha = None
    previous_ok = True
    env = os.environ.copy()
    env["PRE_COMMIT_HOME"] = str(pre_commit_home)
    env["PATH"] = str(_pre_commit_bin().parent) + os.pathsep + env["PATH"]
    for step in steps:
        step_if = step.get("if")
        step_context = {"github": github, "needs": {}}
        if not previous_ok or not _condition(step_if, step_context):
            continue
        uses = str(step.get("uses") or "")
        try:
            if uses.startswith("actions/checkout@"):
                ref, commit, allow = _checkout_inputs(step, github)
                _assert_safe_pr_checkout(
                    github["event_name"],
                    payload,
                    f"{github['repository_owner']}/{github['event']['repository']['name']}",
                    ref,
                    commit,
                    allow,
                )
                if workspace.exists():
                    shutil.rmtree(workspace)
                checkout_sha = _checkout_tree(source, workspace, ref, commit)
            elif "pre-commit/action@" in uses:
                completed = subprocess.run(
                    ["bash", "-lc", _pre_commit_command(step)],
                    cwd=workspace,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output.append(completed.stdout)
                output.append(completed.stderr)
            elif "run" in step:
                completed = subprocess.run(
                    ["bash", "-lc", str(step["run"])],
                    cwd=workspace if workspace.exists() else None,
                    env=env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                output.append(completed.stdout)
                output.append(completed.stderr)
                if completed.returncode != 0:
                    previous_ok = False
        except (RuntimeError, subprocess.CalledProcessError) as error:
            output.append(str(error))
            previous_ok = False
    return checkout_sha, "".join(output)


def _github_context(scenario: Scenario, trees: dict[str, str]) -> tuple[dict, dict, dict]:
    pr = {
        "number": 1,
        "user": {"login": "contributor"},
        "labels": [{"name": label} for label in scenario.labels],
        "head": {
            "sha": trees["head"],
            "repo": {"id": 2, "full_name": "contributor/vllm"},
        },
        "base": {
            "sha": trees["base"],
            "ref": "main",
            "repo": {"id": 1, "full_name": f"{scenario.owner}/vllm"},
        },
        "merge_commit_sha": trees["merge"],
    }
    payload = {
        "action": scenario.action,
        "number": 1,
        "repository": {
            "id": 1,
            "name": "vllm",
            "default_branch": trees["default_branch"],
        },
        "pull_request": pr,
    }
    github = {
        "repository_owner": scenario.owner,
        "repository": f"{scenario.owner}/vllm",
        "event_name": scenario.event_name,
        "ref": "refs/pull/1/merge",
        "sha": trees["merge"],
        "event": payload,
    }
    script_context = {
        "repo": {"owner": scenario.owner, "repo": "vllm"},
        "payload": {"pull_request": {"number": pr["number"]}},
        "eventName": scenario.event_name,
    }
    return github, payload, script_context


def replay(scenario: Scenario, tmp: Path) -> Replay:
    marker = tmp / "marker"
    trees = _prepare_repo(tmp, marker, scenario)
    workflow = _load_workflow()
    github, payload, script_context = _github_context(scenario, trees)
    if not _triggered(workflow, scenario.event_name, scenario.action, github["ref"]):
        return Replay(marker, False, "workflow not triggered", None)

    jobs = workflow["jobs"]
    needs: dict[str, dict[str, str]] = {}
    check = jobs.get("pre-run-check")
    if isinstance(check, dict):
        check_context = {"github": github, "needs": needs}
        if _condition(check.get("if"), check_context):
            script = _gate_script(workflow)
            if script is None:
                needs["pre-run-check"] = {"result": "success"}
            else:
                failed = _run_gate_script(
                    script,
                    script_context,
                    payload["pull_request"],
                    scenario.merged_count,
                )
                needs["pre-run-check"] = {
                    "result": "failure" if failed else "success"
                }
        else:
            needs["pre-run-check"] = {"result": "skipped"}

    job = jobs.get("pre-commit")
    if not isinstance(job, dict):
        return Replay(marker, False, "pre-commit job absent", None)
    job_context = {"github": github, "needs": needs}
    if not _condition(job.get("if"), job_context):
        return Replay(marker, False, "pre-commit job skipped", None)
    labels = _runs_on_labels(job.get("runs-on"), job_context)
    if not _self_hosted(labels):
        return Replay(marker, False, f"runs-on={labels}", None)

    workspace = tmp / "workspace"
    checkout_sha, output = _run_steps(
        job.get("steps", []),
        source=trees["source"],
        workspace=workspace,
        github=github,
        payload=payload,
        pre_commit_home=tmp / "pre-commit-home",
    )
    return Replay(marker, True, output, checkout_sha)


SCENARIOS = [
    Scenario("opened-four-merged-explicit-manual", 4, (), hook="local-manual"),
    Scenario(
        "opened-four-merged-builtin-stages",
        4,
        (),
        hook="local-builtin-stages",
    ),
    Scenario(
        "opened-four-merged-repo-default-stages",
        4,
        (),
        hook="local-repo-default-stages",
    ),
    Scenario("synchronize-four-merged", 4, (), action="synchronize"),
    Scenario("reopened-four-merged", 4, (), action="reopened"),
    Scenario("labeled-ready", 0, ("ready",), action="labeled"),
    Scenario("opened-ready", 3, ("ready",)),
    Scenario("opened-verified", 0, ("verified",)),
    Scenario("opened-ready-run-all-tests", 0, ("ready-run-all-tests",)),
    Scenario("opened-four-merged-remote-rev", 4, (), hook="remote-rev"),
    Scenario("opened-below-threshold", 3, (), hook="local-manual"),
    Scenario("opened-unrelated-label", 0, ("bug",), hook="local-manual"),
    Scenario(
        "opened-other-owner",
        4,
        ("ready",),
        owner="example-owner",
        hook="local-manual",
    ),
]


def _pre_commit_bin() -> Path:
    return Path(sys.executable).parent / "pre-commit"


@pytest.mark.parametrize("scenario", SCENARIOS, ids=lambda scenario: scenario.name)
def test_pull_request_hooks_do_not_execute_on_self_hosted_runners(
    scenario: Scenario, tmp_path: Path
) -> None:
    """A pull request must not run its hook command on a self-hosted runner.

    The gate is the workflow's own script. Re-checking the same author or
    label and then checking out the pull request merge commit still fails
    this test, because the marker is written by that commit's hook.
    """

    assert _pre_commit_bin().is_file(), (
        "install pre-commit==4.7.0 next to the test interpreter"
    )
    result = replay(scenario, tmp_path)
    assert not result.marker.exists(), result.message()
