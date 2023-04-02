# managing-ai

Edit the markdown files, or run jupyter lab to edit the notebooks

```bash
jupyter lab book
```

To build the book, do this from the project root:

```bash
jupyter-book build book

git add book/_build
git commit -m "..."
git push
```

## One-time setup

ghp-import -n -p -f book/_build/html
