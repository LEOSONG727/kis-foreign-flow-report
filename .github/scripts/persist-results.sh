#!/usr/bin/env bash
# Usage: bash .github/scripts/persist-results.sh "commit message" path ...
set -euo pipefail
message="${1:?A commit message is required}"
shift
target="${GITHUB_REF_NAME:?GITHUB_REF_NAME is required}"
if [[ "${GITHUB_REF_TYPE:-branch}" != "branch" ]]; then
  echo "::error::Results can only be persisted to a branch, not a tag."
  exit 1
fi
git check-ref-format "refs/heads/$target"
git config user.name "foreign-flow-bot"
git config user.email "foreign-flow-bot@users.noreply.github.com"
if ! git diff --cached --quiet; then
  echo "::error::Unexpected staged files before persisting results."
  exit 1
fi
for path in "$@"; do
  if [[ -e "$path" ]]; then git add -- "$path"; fi
done
if git diff --cached --quiet; then
  echo "변경된 결과가 없습니다."
  exit 0
fi
git commit -m "$message"
mkdir -p logs
git format-patch -1 HEAD --stdout > logs/persist-results.patch
for attempt in 1 2 3 4 5; do
  if git pull --rebase --autostash origin "$target"; then
    if git push origin "HEAD:refs/heads/$target"; then
      rm -f logs/persist-results.patch
      exit 0
    fi
  else
    # A content conflict is not solved by repeating the same rebase, nor by
    # overwriting newer data. Abort and retain the outputs and recovery patch.
    if [[ -n "$(git diff --name-only --diff-filter=U)" ]] ||
       [[ -d "$(git rev-parse --git-path rebase-merge)" ]] ||
       [[ -d "$(git rev-parse --git-path rebase-apply)" ]]; then
      git rebase --abort 2>/dev/null || true
      echo "::error::Result files conflict with remote changes. Recover from the uploaded results/patch."
      exit 1
    fi
  fi
  if (( attempt < 5 )); then sleep $((attempt * 15)); fi
done
echo "::error::Push failed after 5 attempts; results and recovery patch are retained."
exit 1
