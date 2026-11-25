# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""A maintainer comment must not build a head pushed after that comment.

`/ci run` and the other comment commands authorize the commenter, then read the
pull request and build whatever head that read returns. These tests pin the
head that existed when the comment was created and fail if Buildkite is asked
to build, or the branch updater is asked to merge, a later head.

Re-reading the pull request does not satisfy them. In the attack cases every
`get_pr` returns a head whose push is newer than `comment.created_at`, and one
case changes that head again on a third read. Git committer dates do not
satisfy them either: `test_backdated_force_push_after_comment_is_not_built`
sets the git committer date before the comment and records the push with
GitHub's own timestamps.

Building nothing is acceptable. Building the pre-comment head is acceptable.
`/ci cancel` and `/amd-ci cancel` only cancel by branch, and `/amd-ci retry`
does not create a build when the new head has none, so those commands are not
asserted here.

The stand-in HTTP transport answers the GitHub and Buildkite calls the script
makes today, plus the reads a fix needs:

- `GET /repos/vllm-project/vllm/pulls/42` is the live head, not the head at
  comment time.
- `GET /repos/vllm-project/vllm/pulls/42/commits` and
  `GET /repos/vllm-project/vllm/commits/{sha}` carry git committer dates.
- `GET /repos/vllm-project/vllm/issues/42/timeline` carries `committed` events
  and `head_ref_force_pushed` events. Force-push `created_at` is GitHub's
  timestamp. `commit_id` is the new head.
- `GET /repos/vllm-project/vllm/issues/42/events` carries the same
  `head_ref_force_pushed` events.
- `GET /repos/vllm-project/vllm/events` carries `PullRequestEvent` `synchronize`
  payloads. `created_at` is when GitHub recorded the push. `before` and `after`
  are the heads around that push, and `payload.pull_request.head.sha` is
  `after`.
- `GET /repos/vllm-project/vllm/issues/comments/99` is the triggering comment,
  including `created_at`. The same timestamp is on the event passed to `run`.
"""

import json
import os
import tempfile
import unittest
import urllib.parse
from typing import Any
from unittest.mock import patch

from run_ci_command import (
    COMMAND_RETRY_FAILED,
    COMMAND_RUN_AMD_CI,
    COMMAND_RUN_CI,
    RUN_CI_COMMAND_ENV,
    ApiError,
    BuildkiteClient,
    GitHubClient,
    main,
    run,
)

SEEN = "a" * 40
SWAPPED = "b" * 40
SWAPPED_AGAIN = "c" * 40
MERGED = "d" * 40
SEEN_AT = "2026-07-28T11:00:00Z"
COMMENT_AT = "2026-07-28T12:00:00Z"
SWAPPED_AT = "2026-07-28T12:00:30Z"
AGAIN_AT = "2026-07-28T12:01:00Z"
BACKDATED_AT = "2020-01-01T00:00:00Z"
FORK_URL = "https://github.com/contributor/vllm.git"
REPO = "vllm-project/vllm"
PR_NUMBER = 42
COMMENT_ID = 99


def _commit_obj(sha: str, committed_at: str) -> dict[str, Any]:
    person = {"name": "Outsider", "email": "outsider@example.com", "date": committed_at}
    return {
        "sha": sha,
        "commit": {
            "message": "update",
            "author": dict(person),
            "committer": dict(person),
        },
    }


class Push:
    def __init__(
        self,
        sha: str,
        pushed_at: str,
        *,
        before: str,
        committed_at: str | None = None,
        force: bool = False,
    ) -> None:
        self.sha = sha
        self.pushed_at = pushed_at
        self.before = before
        self.committed_at = pushed_at if committed_at is None else committed_at
        self.force = force


class World:
    """One scripted pull request. `heads` are successive live `get_pr` results.

    The last head repeats. `pushes` are the only record of when each head
    landed. `authorized` is the head whose push time is at or before the
    comment; a build of any other commit is a failure.
    """

    def __init__(
        self,
        heads: list[str],
        pushes: list[Push],
        *,
        authorized: str = SEEN,
        behind: dict[str, int] | None = None,
        previous_builds: list[dict[str, Any]] | None = None,
        author_permission: str = "none",
    ) -> None:
        self.heads = heads
        self.pushes = pushes
        self.authorized = authorized
        self.behind = behind or {}
        self.previous_builds = previous_builds or []
        self.author_permission = author_permission
        self.pull_reads = 0
        self.head_override: str | None = None
        self.builds: list[dict[str, Any]] = []
        self.updates: list[str] = []

    def live_sha(self) -> str:
        if self.head_override is not None:
            return self.head_override
        index = min(self.pull_reads, len(self.heads) - 1)
        return self.heads[index]

    def pull(self) -> dict[str, Any]:
        sha = self.live_sha()
        self.pull_reads += 1
        return {
            "base": {"ref": "main"},
            "draft": False,
            "head": {
                "label": "outsider:feature",
                "ref": "feature",
                "repo": {"clone_url": FORK_URL},
                "sha": sha,
            },
            "labels": [],
            "number": PR_NUMBER,
            "state": "open",
            "user": {"login": "outsider"},
        }

    def commit_record(self, sha: str) -> dict[str, Any] | None:
        for push in self.pushes:
            if push.sha == sha:
                return _commit_obj(sha, push.committed_at)
        return None


class Transport:
    def __init__(self, world: World) -> None:
        self.world = world

    def request(
        self,
        url: str,
        *,
        body: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        method: str = "GET",
    ) -> Any:
        del headers
        parts = urllib.parse.urlsplit(url)
        path = urllib.parse.unquote(parts.path)
        query = urllib.parse.parse_qs(parts.query)
        if parts.netloc == "api.github.com":
            return self._github(method, path, query, body)
        if parts.netloc == "api.buildkite.com":
            return self._buildkite(method, path, query, body)
        raise ApiError(None, f"unexpected host in test transport: {url}")

    def _page(self, query: dict[str, list[str]], items: list[Any]) -> list[Any]:
        page = int(query.get("page", ["1"])[0])
        if page > 1:
            return []
        return items

    def _github(
        self,
        method: str,
        path: str,
        query: dict[str, list[str]],
        body: dict[str, Any] | None,
    ) -> Any:
        world = self.world
        prefix = f"/repos/{REPO}"
        if path == "/graphql":
            return {
                "data": {
                    "repository": {
                        "pullRequest": {"reviewDecision": "REVIEW_REQUIRED"}
                    }
                }
            }
        if path == f"{prefix}/issues/comments/{COMMENT_ID}/reactions":
            return [] if method == "GET" else {}
        if path == f"{prefix}/issues/comments/{COMMENT_ID}":
            return {
                "id": COMMENT_ID,
                "body": "/ci run",
                "created_at": COMMENT_AT,
                "user": {"login": "reviewer"},
            }
        if path == f"{prefix}/issues/{PR_NUMBER}/comments":
            return [] if method == "GET" else {}
        if path == f"{prefix}/pulls/{PR_NUMBER}/update-branch":
            expected = str((body or {}).get("expected_head_sha", ""))
            world.updates.append(expected)
            world.head_override = MERGED
            world.behind.setdefault(MERGED, 0)
            return {}
        if path == f"{prefix}/pulls/{PR_NUMBER}/commits":
            records = [
                record
                for push in world.pushes
                if (record := world.commit_record(push.sha)) is not None
            ]
            return self._page(query, records)
        if path == f"{prefix}/pulls/{PR_NUMBER}":
            return world.pull()
        if path.startswith(f"{prefix}/commits/"):
            sha = path.rsplit("/", 1)[1]
            record = world.commit_record(sha)
            if record is None:
                raise ApiError(404, "Not found")
            return record
        if path.startswith(f"{prefix}/compare/"):
            sha = path.rsplit("...", 1)[1]
            return {"behind_by": world.behind.get(sha, 0)}
        if path.startswith(f"{prefix}/collaborators/") and path.endswith(
            "/permission"
        ):
            user = path.split("/")[5]
            permission = "write" if user == "reviewer" else world.author_permission
            return {"permission": permission}
        if path == f"{prefix}/issues/{PR_NUMBER}/timeline":
            return self._page(query, self._timeline(world))
        if path == f"{prefix}/issues/{PR_NUMBER}/events":
            issue_events = [
                event
                for event in self._timeline(world)
                if event["event"] != "committed"
            ]
            return self._page(query, issue_events)
        if path == f"{prefix}/events":
            return self._page(query, self._repository_events(world))
        raise ApiError(None, f"unexpected GitHub URL in test transport: {path}")

    def _timeline(self, world: World) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        for push in world.pushes:
            commit = world.commit_record(push.sha)
            assert commit is not None
            events.append(
                {
                    "event": "committed",
                    "sha": push.sha,
                    "author": commit["commit"]["author"],
                    "committer": commit["commit"]["committer"],
                    "message": commit["commit"]["message"],
                }
            )
            if push.force:
                events.append(
                    {
                        "id": 1,
                        "event": "head_ref_force_pushed",
                        "commit_id": push.sha,
                        "created_at": push.pushed_at,
                        "actor": {"login": "outsider"},
                    }
                )
        return events

    def _repository_events(self, world: World) -> list[dict[str, Any]]:
        events = []
        for push in world.pushes:
            events.append(
                {
                    "id": push.sha[:8],
                    "type": "PullRequestEvent",
                    "created_at": push.pushed_at,
                    "payload": {
                        "action": "synchronize",
                        "number": PR_NUMBER,
                        "before": push.before,
                        "after": push.sha,
                        "pull_request": {
                            "number": PR_NUMBER,
                            "head": {"ref": "feature", "sha": push.sha, "repo": {}},
                            "base": {"ref": "main", "sha": "e" * 40, "repo": {}},
                        },
                    },
                }
            )
        return events

    def _buildkite(
        self,
        method: str,
        path: str,
        query: dict[str, list[str]],
        body: dict[str, Any] | None,
    ) -> Any:
        world = self.world
        if path.endswith("/retry_failed_jobs"):
            return {"retried_jobs_count": 1}
        if path.endswith("/jobs"):
            return {
                "items": [
                    {
                        "state": "failed",
                        "step_key": "unit-test",
                        "type": "script",
                    }
                ],
                "links": {"next": None},
            }
        if not path.endswith("/builds"):
            raise ApiError(None, f"unexpected Buildkite URL in test transport: {path}")
        if method == "POST":
            assert body is not None
            world.builds.append(dict(body))
            return {
                "number": 123,
                "web_url": "https://buildkite.example/builds/123",
            }
        if any(key.startswith("meta_data[github-pr-number]") for key in query):
            return world.previous_builds
        return []


def _event(command: str, actor: str = "reviewer") -> dict[str, Any]:
    return {
        "action": "created",
        "comment": {
            "body": command,
            "created_at": COMMENT_AT,
            "id": COMMENT_ID,
            "user": {"login": actor},
        },
        "issue": {"number": PR_NUMBER, "pull_request": {}},
    }


def _clients(world: World, pipeline: str) -> tuple[GitHubClient, BuildkiteClient]:
    token = "placeholder-token"
    transport = Transport(world)
    return (
        GitHubClient(token, REPO, transport),
        BuildkiteClient(token, "vllm", pipeline, transport),
    )


def _updater(world: World, github: GitHubClient) -> GitHubClient:
    return GitHubClient("placeholder-token", REPO, github.transport)


def _seen_push() -> Push:
    return Push(SEEN, SEEN_AT, before="0" * 40)


def _attack_pushes(force: bool = False, committed_at: str | None = None) -> list[Push]:
    return [
        _seen_push(),
        Push(
            SWAPPED,
            SWAPPED_AT,
            before=SEEN,
            committed_at=committed_at,
            force=force,
        ),
    ]


class CiCommentHeadPinTest(unittest.TestCase):
    def assert_only_pre_comment_head(self, world: World) -> None:
        built = [build["commit"] for build in world.builds]
        bad_builds = [commit for commit in built if commit != world.authorized]
        bad_updates = [sha for sha in world.updates if sha != world.authorized]
        self.assertEqual(
            (bad_builds, bad_updates),
            ([], []),
            "acted on a head pushed after the comment; "
            f"authorized head is {world.authorized}",
        )

    def _run(
        self,
        world: World,
        command: str,
        *,
        actor: str = "reviewer",
        updater: bool = False,
    ) -> None:
        pipeline = "amd-ci" if command.startswith("/amd-ci") else "ci"
        github, buildkite = _clients(world, pipeline)
        branch_updater = _updater(world, github) if updater else None
        with patch("run_ci_command.time.sleep"):
            run(
                _event(command, actor),
                github,
                buildkite,
                "",
                branch_updater,
            )

    def test_head_pushed_after_comment_is_not_built_when_reread_is_stable(
        self,
    ) -> None:
        for command in sorted(RUN_CI_COMMAND_ENV):
            with self.subTest(command=command):
                world = World([SWAPPED], _attack_pushes())
                self._run(world, command, updater=True)
                self.assert_only_pre_comment_head(world)

    def test_later_reread_of_the_same_pull_request_is_not_built(self) -> None:
        world = World([SWAPPED, SWAPPED, SWAPPED_AGAIN], _attack_pushes())
        world.pushes.append(Push(SWAPPED_AGAIN, AGAIN_AT, before=SWAPPED))
        self._run(world, COMMAND_RUN_CI, updater=True)
        self.assert_only_pre_comment_head(world)

    def test_backdated_force_push_after_comment_is_not_built(self) -> None:
        world = World(
            [SWAPPED],
            _attack_pushes(force=True, committed_at=BACKDATED_AT),
        )
        self._run(world, COMMAND_RUN_CI, updater=True)
        self.assert_only_pre_comment_head(world)

    def test_outdated_head_pushed_after_comment_is_not_merged_or_built(
        self,
    ) -> None:
        for command in (COMMAND_RUN_CI, COMMAND_RUN_AMD_CI):
            with self.subTest(command=command):
                world = World(
                    [SWAPPED],
                    _attack_pushes(),
                    behind={SWAPPED: 4},
                )
                self._run(world, command, updater=True)
                self.assert_only_pre_comment_head(world)

    def test_retry_does_not_build_head_pushed_after_comment(self) -> None:
        world = World(
            [SWAPPED],
            _attack_pushes(),
            previous_builds=[
                {
                    "commit": SEEN,
                    "created_at": SEEN_AT,
                    "finished_at": SEEN_AT,
                    "number": 122,
                    "pull_request": {"id": PR_NUMBER},
                    "state": "failed",
                    "web_url": "https://buildkite.example/builds/122",
                }
            ],
        )
        self._run(world, COMMAND_RETRY_FAILED)
        self.assert_only_pre_comment_head(world)

    def test_maintainer_run_builds_head_that_predates_the_comment(self) -> None:
        for command in (COMMAND_RUN_CI, f"{COMMAND_RUN_AMD_CI} --allow-stale"):
            with self.subTest(command=command):
                world = World([SEEN], [_seen_push()])
                self._run(world, command, updater=True)
                self.assertEqual([build["commit"] for build in world.builds], [SEEN])
                self.assertEqual(
                    world.builds[0]["pull_request_repository"],
                    FORK_URL,
                )
                self.assertEqual(world.updates, [])

    def test_maintainer_run_still_merges_an_outdated_pre_comment_head(self) -> None:
        world = World([SEEN], [_seen_push()], behind={SEEN: 4, MERGED: 0})
        self._run(world, COMMAND_RUN_CI, updater=True)
        self.assertEqual(world.updates, [SEEN])
        self.assertEqual([build["commit"] for build in world.builds], [MERGED])

    def test_author_without_delegation_cannot_start_ci_on_the_new_head(self) -> None:
        world = World([SWAPPED], _attack_pushes())
        self._run(world, COMMAND_RUN_CI, actor="outsider")
        self.assertEqual(world.builds, [])
        self.assertEqual(world.updates, [])

    def test_issue_comment_entrypoint_does_not_build_a_post_comment_head(
        self,
    ) -> None:
        world = World([SWAPPED], _attack_pushes())
        transport = Transport(world)

        class BoundTransport:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                del args, kwargs

            def request(self, url: str, **kwargs: Any) -> Any:
                return transport.request(url, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            event_path = os.path.join(directory, "event.json")
            with open(event_path, "w", encoding="utf-8") as event_file:
                json.dump(_event(COMMAND_RUN_CI), event_file)
            env = {
                "GITHUB_EVENT_PATH": event_path,
                "GITHUB_EVENT_NAME": "issue_comment",
                "GITHUB_REPOSITORY": REPO,
                "GH_TOKEN": "placeholder-token",
                "GH_BRANCH_UPDATE_TOKEN": "placeholder-token",
                "BUILDKITE_API_TOKEN": "placeholder-token",
                "BUILDKITE_ORGANIZATION": "vllm",
                "BUILDKITE_PIPELINE": "ci",
                "BUILDKITE_AMD_PIPELINE": "amd-ci",
                "CI_TRUSTED_USERS": "",
            }
            with (
                patch("run_ci_command.HttpTransport", BoundTransport),
                patch("run_ci_command.time.sleep"),
                patch.dict(os.environ, env, clear=False),
            ):
                main()
        self.assert_only_pre_comment_head(world)


if __name__ == "__main__":
    unittest.main()
