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

For newly published paper blogs, the Markdown article itself must be written as two explicit parts:

1. A top summary card content section with concrete authored analysis.
2. The formal long-form paper analysis body.

Do not rely on `excerpt`, title headings, or metadata to automatically synthesize this card. The website may render the authored opening section as card UI, but the source content must exist in the Markdown post itself. Use this Markdown-compatible structure before the long-form explanation:

```markdown
## ✦ 核心洞察与挑战

### 核心问题

### 传统方案局限

## 研究动机

## 方法论（主要模块简介）
```

Keep this opening section concise and card-friendly. Put detailed experiments, figures, and numeric analysis in later sections unless a key number is necessary for the opening thesis.

Before publishing a paper blog, check `posts/index.json` and existing Markdown files under `posts/` for the same paper title, DOI, arXiv ID, or obvious slug match. If the paper has already been published, remind the user that it is already published and do not publish a duplicate. Only update, rewrite, or replace the existing post when the user explicitly asks to update, rewrite, republish, or overwrite it.

## Python Environment

Use the current project's uv-managed Python environment for Python work.

- Add Python dependencies with `uv add <package>`.
- Run Python scripts with `uv run python ...`.
- Do not use ad-hoc temporary virtual environments for project work.
- The current PDF/image processing dependencies are managed in `pyproject.toml`: `pymupdf` and `pillow`.
