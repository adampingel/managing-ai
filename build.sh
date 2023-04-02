#!/usr/bin/env bash

set -e
set -x

jupyter-book build .

git add _build
git commit -m "..."
git push

ghp-import -n -p -f _build/html

echo "Browse to https://github.com/adampingel/managing-ai/actions to check progress"

echo "Browse to https://adampingel.github.io/managing-ai/ to see the updates live"