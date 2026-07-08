#!/usr/bin/env python3
"""Upload quantized model weights to LakeFS via data_registry."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

from data_registry import Dataset

DEFAULT_CONFIG = "/home/jovyan/isupov/libs/data_registry/new_config.yaml"
DEFAULT_REPO = "idp-agentic-retrieval"
DEFAULT_BRANCH = "main"
DEFAULT_WEIGHTS_DIR = (
    "/home/jovyan/isupov/agentic-retrieval/checkpoints/quantized/qwen3_5_35b_a3b_w8a8"
)
DEFAULT_REMOTE_PREFIX = "qwen3_5_35b_a3b_w8a8"


def _validate_weights_dir(weights_dir: str) -> list[str]:
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"Weights directory not found: {weights_dir}")

    files = [
        name
        for name in os.listdir(weights_dir)
        if os.path.isfile(os.path.join(weights_dir, name))
    ]
    if not files:
        raise ValueError(f"No files found in weights directory: {weights_dir}")

    required = {"config.json", "model.safetensors", "tokenizer.json"}
    missing = sorted(required - set(files))
    if missing:
        raise ValueError(f"Missing expected weight files in {weights_dir}: {missing}")

    return files


def _staging_dir_for_remote_prefix(weights_dir: str, remote_prefix: str) -> str:
    """Build a staging tree so files land under <remote_prefix>/ in LakeFS.

    data_registry walks the sync directory with os.walk, which does not follow
    directory symlinks, so we symlink individual files instead.
    """
    staging_root = tempfile.mkdtemp(prefix="lakefs_weights_upload_")
    remote_dir = os.path.join(staging_root, remote_prefix)
    os.makedirs(remote_dir)

    for name in os.listdir(weights_dir):
        src = os.path.join(weights_dir, name)
        if not os.path.isfile(src):
            continue
        os.symlink(os.path.abspath(src), os.path.join(remote_dir, name))

    return staging_root


def upload_weights(
    *,
    config_path: str,
    repo: str,
    branch: str,
    weights_dir: str,
    remote_prefix: str,
    commit_message: str,
    dry_run: bool = False,
) -> None:
    files = _validate_weights_dir(weights_dir)
    staging_root = _staging_dir_for_remote_prefix(weights_dir, remote_prefix)

    print(f"Repo:           {repo}")
    print(f"Branch:         {branch}")
    print(f"Local weights:  {weights_dir}")
    print(f"Remote prefix:  {remote_prefix}/")
    print(f"Files to sync:  {len(files)}")

    dataset = Dataset(config_path=config_path, repo=repo, branch=branch)
    dataset.sync(staging_root)

    status = dataset.status()
    diff = status["diff"]
    print(
        "Planned diff: "
        f"added={len(diff['added'])}, "
        f"updated={len(diff['updated'])}, "
        f"removed={len(diff['removed'])}"
    )
    if diff["removed"]:
        print(
            "WARNING: sync replaces the full branch snapshot. "
            f"{len(diff['removed'])} existing remote file(s) would be removed."
        )

    if dry_run:
        print("Dry run only; skipping upload/finalize.")
        return

    dataset.upload()
    commit_id = dataset.finalize(commit_message)
    print(f"Committed as {commit_id}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--branch", default=DEFAULT_BRANCH)
    parser.add_argument("--weights-dir", default=DEFAULT_WEIGHTS_DIR)
    parser.add_argument("--remote-prefix", default=DEFAULT_REMOTE_PREFIX)
    parser.add_argument(
        "--commit-message",
        default="Upload qwen3_5_35b_a3b_w8a8 quantized weights",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths and show planned diff without uploading.",
    )
    args = parser.parse_args()

    try:
        upload_weights(
            config_path=args.config,
            repo=args.repo,
            branch=args.branch,
            weights_dir=args.weights_dir,
            remote_prefix=args.remote_prefix,
            commit_message=args.commit_message,
            dry_run=args.dry_run,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
