# AGENTS.md

## Paper Blog Publishing

When the user asks to generate or publish a blog post based on an academic paper, paper URL, arXiv link, DOI, uploaded PDF, or local PDF, use the `paper-blog-analysis` skill.

This includes requests such as:

- Analyze a paper and write a blog post.
- Summarize an arXiv paper and publish it on this site.
- Turn a local PDF into a Chinese paper-reading note.
- Add a paper interpretation article to the blog.

The skill to use is:

```text
paper-blog-analysis
```

Follow the skill's required structure for deep paper analysis and publishable Chinese academic-style blog writing. If the user asks to publish the result into this project, add the article body as a Markdown file under `posts/`, then register its metadata in `posts/index.json`. When selected paper figures/tables help the article, save them under `posts/assets/<post-id>/` and embed them in Markdown with relative paths such as `![图 1：caption](assets/<post-id>/figure-1.png)`. Do not add a mandatory standalone figure-by-figure interpretation section unless the user asks for it. Keep tags suitable for filtering, such as `LLM`, `自动驾驶`, `AI Infra`, `RAG`, or other paper-specific topics.

## Python Environment

Use the current project's uv-managed Python environment for Python work.

- Add Python dependencies with `uv add <package>`.
- Run Python scripts with `uv run python ...`.
- Do not use ad-hoc temporary virtual environments for project work.
- The current PDF/image processing dependencies are managed in `pyproject.toml`: `pymupdf` and `pillow`.
