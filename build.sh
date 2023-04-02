#!/usr/bin/env bash

set -e
set -x

jupyter-book build .

git add _build
git commit -m "..."
git push

ghp-import -n -p -f _build/html
