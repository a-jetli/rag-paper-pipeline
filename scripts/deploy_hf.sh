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

if [ ! -f chroma_db/bm25_index.pickle ] || [ ! -f chroma_db/chroma.sqlite3 ]; then
    echo "chroma_db/ is missing or incomplete. Run scripts/build_index.py first." >&2
    exit 1
fi

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Working tree has uncommitted tracked changes. Commit them to main first." >&2
    exit 1
fi

chunks=$(python3 -c "
import chromadb
print(chromadb.PersistentClient(path='chroma_db').get_collection('arxiv_papers').count())
")

GIT_INDEX_FILE="$(mktemp -t rag-deploy-index)"
export GIT_INDEX_FILE
trap 'rm -f "$GIT_INDEX_FILE"' EXIT

git read-tree main
git add -f chroma_db/

# Everything large must be an LFS pointer. A 576MB blob committed as a plain Git
# object is painful to undo, so fail loudly rather than discover it mid-push.
staged_big=$(git diff --cached --name-only main -- chroma_db | wc -l | tr -d ' ')
lfs_tracked=$(git lfs status | grep -c 'LFS:' || true)
if [ "$lfs_tracked" -lt "$staged_big" ]; then
    echo "Only $lfs_tracked of $staged_big chroma_db files are LFS-tracked. Check .gitattributes." >&2
    exit 1
fi

tree=$(git write-tree)
commit=$(git commit-tree "$tree" -p main -m "deploy: main + prebuilt index (${chunks} chunks)")

echo "Built deploy commit ${commit:0:10} on top of $(git rev-parse --short main) (${chunks} chunks)."
echo "Pushing to the Space. This uploads ~800MB through LFS on a corpus change."
git push -f huggingface "$commit:main"
