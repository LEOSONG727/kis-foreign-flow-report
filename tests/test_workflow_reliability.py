from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
PERSIST = ROOT / ".github/scripts/persist-results.sh"


def git(directory, *args):
    return subprocess.run(["git", *args], cwd=directory, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repositories(tmp_path):
    remote = tmp_path / "remote.git"
    local = tmp_path / "local"
    other = tmp_path / "other"
    git(tmp_path, "init", "--bare", str(remote))
    git(tmp_path, "clone", str(remote), str(local))
    git(local, "switch", "-c", "main")
    for directory in [local]:
        git(directory, "config", "user.name", "test")
        git(directory, "config", "user.email", "test@example.invalid")
    (local / "data").mkdir()
    (local / "data/result.csv").write_text("value\nbase\n")
    (local / "code.txt").write_text("base\n")
    (local / "notes.txt").write_text("base\n")
    git(local, "add", ".")
    git(local, "commit", "-m", "seed")
    git(local, "push", "-u", "origin", "main")
    git(tmp_path, "clone", "--branch", "main", str(remote), str(other))
    git(other, "config", "user.name", "test")
    git(other, "config", "user.email", "test@example.invalid")
    return local, other


def persist(local, *, ref_type="branch"):
    return subprocess.run(["bash", str(PERSIST), "bot: test result", "data/result.csv", "absent.csv"],
                          cwd=local, env={**os.environ, "GITHUB_REF_NAME": "main", "GITHUB_REF_TYPE": ref_type},
                          capture_output=True, text=True, timeout=30)


def test_push_rebases_new_remote_commit_and_autostashes_unrelated_edits(repositories):
    local, other = repositories
    (local / "data/result.csv").write_text("value\nnew-result\n")
    (local / "notes.txt").write_text("unstaged-local-notes\n")
    (other / "code.txt").write_text("remote-code-change\n")
    git(other, "add", "code.txt")
    git(other, "commit", "-m", "concurrent code update")
    git(other, "push", "origin", "main")
    result = persist(local)
    assert result.returncode == 0, result.stdout + result.stderr
    git(other, "pull", "--ff-only")
    assert (other / "data/result.csv").read_text() == "value\nnew-result\n"
    assert (local / "code.txt").read_text() == "remote-code-change\n"
    assert (local / "notes.txt").read_text() == "unstaged-local-notes\n"
    assert (other / "notes.txt").read_text() == "base\n"
    assert not (local / "logs/persist-results.patch").exists()


def test_content_conflict_fails_without_overwriting_and_retains_recovery_patch(repositories):
    local, other = repositories
    (local / "data/result.csv").write_text("value\nlocal-result\n")
    (other / "data/result.csv").write_text("value\nnewer-remote-result\n")
    git(other, "add", "data/result.csv")
    git(other, "commit", "-m", "concurrent result update")
    git(other, "push", "origin", "main")
    result = persist(local)
    assert result.returncode == 1
    assert "::error::Result files conflict" in result.stdout
    assert (local / "data/result.csv").read_text() == "value\nlocal-result\n"
    assert (other / "data/result.csv").read_text() == "value\nnewer-remote-result\n"
    assert (local / "logs/persist-results.patch").exists()
    assert git(local, "diff", "--name-only", "--diff-filter=U") == ""


def test_no_changes_and_tag_dispatch(repositories):
    local, other = repositories
    assert persist(local).returncode == 0
    assert persist(local, ref_type="tag").returncode == 1


def test_autostash_conflict_is_not_reported_as_success(repositories):
    local, other = repositories
    (local / "data/result.csv").write_text("value\nlocal-result\n")
    (local / "notes.txt").write_text("local-notes\n")
    (other / "notes.txt").write_text("remote-notes\n")
    git(other, "add", "notes.txt")
    git(other, "commit", "-m", "concurrent notes update")
    git(other, "push", "origin", "main")
    result = persist(local)
    assert result.returncode == 1
    assert "::error::Autostash has conflicts" in result.stdout
    assert (other / "data/result.csv").read_text() == "value\nbase\n"
    assert (local / "logs/persist-results.patch").exists()


def test_all_six_workflows_share_queue_and_persistence():
    files = sorted((ROOT / ".github/workflows").glob("*.yml"))
    assert len(files) == 6
    for path in files:
        # BaseLoader keeps YAML 1.1's 'on' and booleans as strings, like Actions YAML.
        workflow = yaml.load(path.read_text(), Loader=yaml.BaseLoader)
        assert workflow["concurrency"] == {"group": "foreign-flow-data-write", "queue": "max", "cancel-in-progress": "false"}
        for job in workflow["jobs"].values():
            steps = job["steps"]
            checkout = next(step for step in steps if step.get("uses", "").startswith("actions/checkout"))
            assert checkout["with"]["ref"] == "${{ github.ref_name }}"
            assert checkout["with"]["fetch-depth"] == "0"
            save_index = next(i for i, step in enumerate(steps) if "persist-results.sh" in step.get("run", ""))
            artifact_index = next(i for i, step in enumerate(steps) if step.get("uses", "").startswith("actions/upload-artifact"))
            assert artifact_index > save_index
            assert steps[artifact_index]["if"] == "always()"
            for step in steps:
                if "run" in step:
                    subprocess.run(["bash", "-n"], input=step["run"], text=True, check=True)
