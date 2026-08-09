#!/usr/bin/env bash
#
# Publish `main` plus the prebuilt index to the Hugging Face Space.
#
# GitHub gets source only — chroma_db/ is ~800MB and GitHub's free LFS tier is
# 1GB, retained per version. The Space needs the index baked into the image.
#
# The obvious way to do that is a `deploy` branch that tracks chroma_db/. Do not.
# Checking it out and returning to main deletes the whole corpus from the working
# tree, because the files are tracked on one branch and absent on the other. Git
# is right to do that and it is still an 800MB footgun.
#
# So this builds the deploy commit in a temporary index instead. The working tree
# is never read from, written to, or switched. Nothing to restore afterwards.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# --source-only reuses the chroma_db subtree already on the Space instead of the
# local one. Use it whenever only code changed. Merely opening the corpus locally
# rewrites chroma.sqlite3 (see the count below), so the local copy is usually
# byte-different from the deployed one even when the data is identical — and a
# byte difference means a fresh 588MB upload for nothing.
SOURCE_ONLY=0
if [ "${1:-}" = "--source-only" ]; then
    SOURCE_ONLY=1
fi

if [ ! -f chroma_db/bm25_index.pickle ] || [ ! -f chroma_db/chroma.sqlite3 ]; then
    echo "chroma_db/ is missing or incomplete. Run scripts/build_index.py first." >&2
    exit 1
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Working tree has uncommitted tracked changes. Commit them to main first." >&2
    exit 1
fi

# Read the count straight out of SQLite in read-only mode. Opening the collection
# through Chroma rewrites chroma.sqlite3 — verified: the file's sha256 changes on
# a bare .count(). Since the LFS object ID *is* that hash, letting Chroma touch
# the file here would mint a byte-different 588MB blob on every single deploy and
# re-upload the whole thing. Anything that opens the corpus before a push costs
# you 588MB of remote storage.
chunks=$(python3 -c "
import sqlite3
c = sqlite3.connect('file:chroma_db/chroma.sqlite3?mode=ro', uri=True)
print(c.execute('SELECT COUNT(*) FROM embeddings').fetchone()[0])
")

GIT_INDEX_FILE="$(mktemp -t rag-deploy-index)"
export GIT_INDEX_FILE
trap 'rm -f "$GIT_INDEX_FILE"' EXIT

git read-tree main

if [ "$SOURCE_ONLY" = "1" ]; then
    git fetch -q huggingface main
    if ! git rev-parse -q --verify FETCH_HEAD:chroma_db >/dev/null; then
        echo "The Space has no chroma_db to reuse. Deploy without --source-only." >&2
        exit 1
    fi
    # Graft the corpus the Space already holds. Its LFS pointers are objects the
    # remote already stores, so this uploads nothing.
    git read-tree --prefix=chroma_db/ FETCH_HEAD:chroma_db
    echo "Reusing the corpus already on the Space; only source files will upload."
else
    git add -f chroma_db/
    # Everything large must be an LFS pointer. A 588MB blob committed as a plain
    # Git object is painful to undo, so fail loudly rather than find out mid-push.
    staged_big=$(git diff --cached --name-only main -- chroma_db | wc -l | tr -d ' ')
    lfs_tracked=$(git lfs status | grep -c 'LFS:' || true)
    if [ "$lfs_tracked" -lt "$staged_big" ]; then
        echo "Only $lfs_tracked of $staged_big chroma_db files are LFS-tracked. Check .gitattributes." >&2
        exit 1
    fi
fi

# The commit is deliberately parentless. The Space does not need history, and
# carrying it actively breaks: HF validates that every LFS pointer in the pushed
# history resolves to an object it still stores, so once an old corpus is deleted
# to free space, any history referencing it is unpushable. An orphan commit also
# means the Space never accumulates storage from superseded corpora.
tree=$(git write-tree)
commit=$(git commit-tree "$tree" -m "deploy: $(git rev-parse --short main) + prebuilt index (${chunks} chunks)")

echo "Built orphan deploy commit ${commit:0:10} from main @ $(git rev-parse --short main) (${chunks} chunks)."
echo "Pushing to the Space. This uploads ~800MB through LFS on a corpus change."
git push -f huggingface "$commit:main"
