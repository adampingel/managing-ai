# managing-ai

A book published to [https://adampingel.github.io/managing-ai/](https://adampingel.github.io/managing-ai/)

A set of Markdown files and Jupyter Notebooks build with Jupyter Book
and published to GitHub Pages.

## Editing Jupyter Notebooks

```bash
jupyter lab
```

## Local builds

To build the book locally, do this from the project root:

```bash
jupyter-book clean .
jupyter-book build .
```

## Publishing

A GitHub action in `.github/workflows/book_gh_pages.yml` publishes the book to GH Pages.
It is configured to run for all commits to `main`.

See the [run log](https://github.com/adampingel/managing-ai/actions/workflows/book_gh_pages.yml)

## Publishing from local environment

```bash
ghp-import -n -p -f _build/html
```
