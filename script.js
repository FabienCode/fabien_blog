const starterPosts = [
  {
    id: "rag-paper-note",
    title: "RAG 论文阅读笔记：检索增强生成为什么有效",
    category: "论文",
    tags: ["LLM", "AI Infra", "RAG", "论文精读"],
    date: "2026-05-05",
    excerpt: "从问题背景、模型结构和实验设计三个角度整理 RAG 的核心贡献。",
    content: `## 背景
大模型的参数记忆很强，但知识更新、事实引用和长尾问题仍然受限。RAG 的核心想法是把生成模型和外部检索记忆组合起来，让答案能够依赖可更新的语料。

## 方法
- 先用检索器从文档库中找到相关片段。
- 再把片段作为上下文交给生成模型。
- 训练和推理时都要关注检索质量与生成质量的耦合。

## 我的判断
> RAG 的价值不只是提升正确率，更重要的是把知识来源从模型参数中解耦出来。

后续自己做技术博客检索时，可以把每篇 Markdown 笔记切成小块，再为标题、标签和正文分别建立索引。`
  },
  {
    id: "bevfusion-paper-note",
    title: "BEVFusion 论文阅读笔记：多传感器自动驾驶感知",
    category: "论文",
    tags: ["自动驾驶", "多模态", "感知", "论文精读"],
    date: "2026-05-02",
    excerpt: "从 BEV 表示、相机与 LiDAR 融合方式整理自动驾驶感知论文的阅读框架。",
    content: `## 关注问题
自动驾驶感知需要把相机、LiDAR 等传感器信息放到统一空间中比较和融合。BEV 表示的优势是更接近规划和控制所需的鸟瞰空间。

## 阅读重点
- 输入模态分别提供什么信息。
- 融合发生在特征层、空间层还是任务头之前。
- 实验是否覆盖检测、分割和鲁棒性场景。

## 我的判断
自动驾驶论文适合固定记录传感器配置、坐标变换、实时性和数据集设置，否则很难横向比较不同方法。`
  },
  {
    id: "leetcode-dp-template",
    title: "LeetCode 动态规划题目的记录模板",
    category: "LeetCode",
    tags: ["DP", "刷题", "模板"],
    date: "2026-04-30",
    excerpt: "用状态定义、转移方程和边界条件组织刷题笔记。",
    content: `## 记录结构
刷动态规划时，不要一上来写代码。先把题目转成几个稳定问题：

- 状态表示什么？
- 最后一步是什么？
- 转移来自哪里？
- 初始化和遍历顺序是什么？

## Python 模板
\`\`\`python
def solve(nums):
    dp = [0] * len(nums)
    for i, value in enumerate(nums):
        dp[i] = value
    return max(dp)
\`\`\`

## 复盘
每道题最后补一句：这题和哪类题相似，下一次应该从哪个角度识别。`
  },
  {
    id: "python-context-manager",
    title: "Python context manager 的实际用途",
    category: "Python",
    tags: ["Python", "工程实践"],
    date: "2026-04-21",
    excerpt: "context manager 不只是文件读写语法糖，也适合管理资源生命周期。",
    content: `## 核心概念
\`with\` 语句负责进入和退出一段受管理的上下文。常见例子是文件、锁、数据库连接和临时配置。

## 示例
\`\`\`python
from contextlib import contextmanager

@contextmanager
def section(name):
    print("enter", name)
    try:
        yield
    finally:
        print("exit", name)
\`\`\`

## 判断
当一段逻辑必须成对执行 setup 和 cleanup 时，就应该考虑 context manager。`
  },
  {
    id: "reading-note",
    title: "读书笔记：把输入变成可复用的判断",
    category: "读书感悟",
    tags: ["阅读", "复盘", "生活"],
    date: "2026-04-12",
    excerpt: "记录不是为了保存原文，而是为了形成之后能调用的判断。",
    content: `## 记录方式
好的读书笔记不应该只是摘抄。更有用的是写清楚三件事：

- 作者的主张是什么？
- 我为什么同意或不同意？
- 这件事会改变我的哪个行为？

> 笔记真正的价值，是让未来的自己能快速恢复当时的思考现场。`
  }
];

const storageKey = "personal-notes-posts";
let posts = loadPosts();
let activeCategory = "全部";
let activeTag = "全部";
let activePostId = posts[0]?.id;

const categoryFilter = document.querySelector("#categoryFilter");
const tagFilterGroup = document.querySelector("#tagFilterGroup");
const tagFilter = document.querySelector("#tagFilter");
const postList = document.querySelector("#postList");
const searchInput = document.querySelector("#searchInput");
const postForm = document.querySelector("#postForm");
const readerCategory = document.querySelector("#readerCategory");
const readerDate = document.querySelector("#readerDate");
const readerTitle = document.querySelector("#readerTitle");
const readerExcerpt = document.querySelector("#readerExcerpt");
const readerTags = document.querySelector("#readerTags");
const readerContent = document.querySelector("#readerContent");
const editorSection = document.querySelector("#editor");
const totalPosts = document.querySelector("#totalPosts");
const totalCategories = document.querySelector("#totalCategories");
const latestDate = document.querySelector("#latestDate");
const recentPosts = document.querySelector("#recentPosts");
const footerTags = document.querySelector("#footerTags");

function loadPosts() {
  const saved = localStorage.getItem(storageKey);
  if (!saved) {
    return starterPosts;
  }

  try {
    const parsed = JSON.parse(saved);
    return Array.isArray(parsed) && parsed.length ? parsed : starterPosts;
  } catch {
    return starterPosts;
  }
}

function savePosts() {
  localStorage.setItem(storageKey, JSON.stringify(posts));
}

function categories() {
  return ["全部", ...new Set(posts.map((post) => post.category))];
}

function tags() {
  if (activeCategory === "全部") {
    return [];
  }

  return [
    "全部",
    ...new Set(
      posts
        .filter((post) => post.category === activeCategory)
        .flatMap((post) => post.tags)
    )
  ];
}

function filteredPosts() {
  const query = searchInput.value.trim().toLowerCase();
  return posts.filter((post) => {
    const categoryMatch = activeCategory === "全部" || post.category === activeCategory;
    const tagMatch = activeTag === "全部" || post.tags.includes(activeTag);
    const text = `${post.title} ${post.category} ${post.tags.join(" ")} ${post.excerpt} ${post.content}`.toLowerCase();
    return categoryMatch && tagMatch && text.includes(query);
  });
}

function renderCategories() {
  categoryFilter.innerHTML = "";
  categories().forEach((category) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = category;
    button.className = category === activeCategory ? "active" : "";
    button.addEventListener("click", () => {
      activeCategory = category;
      activeTag = "全部";
      const first = filteredPosts()[0];
      activePostId = first?.id || activePostId;
      render();
    });
    categoryFilter.append(button);
  });
}

function renderTags() {
  tagFilter.innerHTML = "";
  const availableTags = tags();

  if (!availableTags.length) {
    tagFilterGroup.classList.add("is-hidden");
    activeTag = "全部";
    return;
  }

  tagFilterGroup.classList.remove("is-hidden");
  if (!availableTags.includes(activeTag)) {
    activeTag = "全部";
  }

  availableTags.forEach((tag) => {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = tag;
    button.className = tag === activeTag ? "active" : "";
    button.addEventListener("click", () => {
      activeTag = tag;
      const first = filteredPosts()[0];
      activePostId = first?.id || activePostId;
      render();
    });
    tagFilter.append(button);
  });
}

function renderPostList() {
  const visiblePosts = filteredPosts();
  postList.innerHTML = "";

  if (!visiblePosts.length) {
    const empty = document.createElement("p");
    empty.className = "post-card-meta";
    empty.textContent = "没有找到匹配的笔记。";
    postList.append(empty);
    return;
  }

  if (!visiblePosts.some((post) => post.id === activePostId)) {
    activePostId = visiblePosts[0].id;
  }

  visiblePosts.forEach((post) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `post-card${post.id === activePostId ? " active" : ""}`;
    button.innerHTML = `
      <span class="post-card-meta">${escapeHTML(post.category)} · ${formatDate(post.date)}</span>
      <h3>${escapeHTML(post.title)}</h3>
      <p>${escapeHTML(post.excerpt)}</p>
    `;
    button.addEventListener("click", () => {
      activePostId = post.id;
      render();
    });
    postList.append(button);
  });
}

function renderReader() {
  const post = posts.find((item) => item.id === activePostId) || filteredPosts()[0] || posts[0];
  if (!post) {
    return;
  }

  readerCategory.textContent = post.category;
  readerDate.textContent = formatDate(post.date);
  readerTitle.textContent = post.title;
  readerExcerpt.textContent = post.excerpt;
  readerTags.innerHTML = post.tags
    .map((tag) => `<button type="button" data-reader-tag="${escapeHTML(tag)}">${escapeHTML(tag)}</button>`)
    .join("");
  readerContent.innerHTML = markdownToHTML(post.content);
}

function renderHomeSummary() {
  const latest = posts
    .map((post) => post.date)
    .sort()
    .at(-1);

  totalPosts.textContent = posts.length;
  totalCategories.textContent = new Set(posts.map((post) => post.category)).size;
  latestDate.textContent = latest ? formatDate(latest).slice(5) : "-";
}

function renderBottomBoard() {
  const latestPosts = [...posts]
    .sort((a, b) => new Date(b.date) - new Date(a.date))
    .slice(0, 3);

  recentPosts.innerHTML = latestPosts
    .map(
      (post) => `
        <button type="button" data-recent-post="${escapeHTML(post.id)}">
          <strong>${escapeHTML(post.title)}</strong>
          <span>${escapeHTML(post.category)} · ${formatDate(post.date)}</span>
        </button>
      `
    )
    .join("");

  footerTags.innerHTML = [...new Set(posts.flatMap((post) => post.tags))]
    .slice(0, 12)
    .map((tag) => `<button type="button" data-footer-tag="${escapeHTML(tag)}">${escapeHTML(tag)}</button>`)
    .join("");
}

function render() {
  renderHomeSummary();
  renderBottomBoard();
  renderCategories();
  renderTags();
  renderPostList();
  renderReader();
}

function openEditor() {
  editorSection.classList.remove("is-hidden");
  editorSection.scrollIntoView({ behavior: "smooth" });
}

function closeEditor() {
  editorSection.classList.add("is-hidden");
}

function markdownToHTML(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const html = [];
  let inCode = false;
  let codeLines = [];
  let listOpen = false;
  let quoteLines = [];

  function closeList() {
    if (listOpen) {
      html.push("</ul>");
      listOpen = false;
    }
  }

  function closeQuote() {
    if (quoteLines.length) {
      html.push(`<blockquote>${quoteLines.map((line) => `<p>${parseInline(line)}</p>`).join("")}</blockquote>`);
      quoteLines = [];
    }
  }

  lines.forEach((line) => {
    if (line.trim().startsWith("```")) {
      closeList();
      closeQuote();
      if (inCode) {
        html.push(`<pre><code>${escapeHTML(codeLines.join("\n"))}</code></pre>`);
        codeLines = [];
        inCode = false;
      } else {
        inCode = true;
      }
      return;
    }

    if (inCode) {
      codeLines.push(line);
      return;
    }

    if (!line.trim()) {
      closeList();
      closeQuote();
      return;
    }

    if (line.startsWith(">")) {
      closeList();
      quoteLines.push(line.replace(/^>\s?/, ""));
      return;
    }

    closeQuote();

    if (line.startsWith("### ")) {
      closeList();
      html.push(`<h3>${parseInline(line.slice(4))}</h3>`);
      return;
    }

    if (line.startsWith("## ")) {
      closeList();
      html.push(`<h2>${parseInline(line.slice(3))}</h2>`);
      return;
    }

    if (line.startsWith("# ")) {
      closeList();
      html.push(`<h2>${parseInline(line.slice(2))}</h2>`);
      return;
    }

    if (/^[-*]\s+/.test(line)) {
      if (!listOpen) {
        html.push("<ul>");
        listOpen = true;
      }
      html.push(`<li>${parseInline(line.replace(/^[-*]\s+/, ""))}</li>`);
      return;
    }

    closeList();
    html.push(`<p>${parseInline(line)}</p>`);
  });

  closeList();
  closeQuote();

  if (inCode) {
    html.push(`<pre><code>${escapeHTML(codeLines.join("\n"))}</code></pre>`);
  }

  return html.join("");
}

function parseInline(value) {
  return escapeHTML(value)
    .replace(/`([^`]+)`/g, "<code>$1</code>")
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
}

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatDate(value) {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit"
  }).format(new Date(value));
}

function slugify(value) {
  const slug = value
    .trim()
    .toLowerCase()
    .replace(/[^\p{Script=Han}\w]+/gu, "-")
    .replace(/^-+|-+$/g, "");
  return `${slug || "note"}-${Date.now().toString(36)}`;
}

searchInput.addEventListener("input", () => {
  const first = filteredPosts()[0];
  activePostId = first?.id || activePostId;
  render();
});

document.querySelectorAll("[data-open-editor]").forEach((trigger) => {
  trigger.addEventListener("click", (event) => {
    event.preventDefault();
    openEditor();
  });
});

document.querySelectorAll("[data-close-editor]").forEach((trigger) => {
  trigger.addEventListener("click", (event) => {
    event.preventDefault();
    closeEditor();
    document.querySelector("#library").scrollIntoView({ behavior: "smooth" });
  });
});

readerTags.addEventListener("click", (event) => {
  const button = event.target.closest("[data-reader-tag]");
  if (!button) {
    return;
  }

  activeCategory = readerCategory.textContent || activeCategory;
  activeTag = button.dataset.readerTag;
  const first = filteredPosts()[0];
  activePostId = first?.id || activePostId;
  render();
  document.querySelector("#library").scrollIntoView({ behavior: "smooth" });
});

recentPosts.addEventListener("click", (event) => {
  const button = event.target.closest("[data-recent-post]");
  if (!button) {
    return;
  }

  const post = posts.find((item) => item.id === button.dataset.recentPost);
  if (!post) {
    return;
  }

  activeCategory = "全部";
  activeTag = "全部";
  activePostId = post.id;
  render();
  document.querySelector("#library").scrollIntoView({ behavior: "smooth" });
});

footerTags.addEventListener("click", (event) => {
  const button = event.target.closest("[data-footer-tag]");
  if (!button) {
    return;
  }

  const tag = button.dataset.footerTag;
  const post = posts.find((item) => item.tags.includes(tag));
  if (!post) {
    return;
  }

  activeCategory = post.category;
  activeTag = tag;
  activePostId = post.id;
  render();
  document.querySelector("#library").scrollIntoView({ behavior: "smooth" });
});

postForm.addEventListener("submit", (event) => {
  event.preventDefault();
  const formData = new FormData(postForm);
  const post = {
    id: slugify(formData.get("title")),
    title: formData.get("title").trim(),
    category: formData.get("category"),
    tags: formData
      .get("tags")
      .split(",")
      .map((tag) => tag.trim())
      .filter(Boolean),
    excerpt: formData.get("excerpt").trim(),
    content: formData.get("content").trim(),
    date: new Date().toISOString().slice(0, 10)
  };

  posts = [post, ...posts];
  activeCategory = "全部";
  activeTag = "全部";
  activePostId = post.id;
  savePosts();
  postForm.reset();
  render();
  closeEditor();
  document.querySelector("#library").scrollIntoView({ behavior: "smooth" });
});

render();
