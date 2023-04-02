# managing-ai

Edit the markdown files, or run jupyter lab to edit the notebooks

```bash
jupyter lab
```

To build the book, do this from the project root:

```bash
jupyter-book build .

git add _build
git commit -m "..."
git push
```

## One-time setup

ghp-import -n -p -f _build/html
