const postsIndexPath = "posts/index.json";
const storageKey = "personal-notes-posts";
let posts = [];
let activeCategory = "全部";
let activeTag = "全部";
let activePostId;

const categoryFilter = document.querySelector("#categoryFilter");
const tagFilterGroup = document.querySelector("#tagFilterGroup");
const tagFilter = document.querySelector("#tagFilter");
const postList = document.querySelector("#postList");
const searchInput = document.querySelector("#searchInput");
const drawerSearchInput = document.querySelector("#drawerSearchInput");
const drawerCategoryList = document.querySelector("#drawerCategoryList");
const drawerTagList = document.querySelector("#drawerTagList");
const drawerPostList = document.querySelector("#drawerPostList");
const sideDrawer = document.querySelector("#sideDrawer");
const readerPage = document.querySelector("#readerPage");
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

async function loadBuiltInPosts() {
  const indexResponse = await fetch(postsIndexPath);
  if (!indexResponse.ok) {
    throw new Error(`Failed to load ${postsIndexPath}`);
  }

  const index = await indexResponse.json();
  return Promise.all(
    index.map(async (post) => {
      const contentResponse = await fetch(`posts/${post.file}`);
      if (!contentResponse.ok) {
        throw new Error(`Failed to load posts/${post.file}`);
      }

      return {
        ...post,
        content: await contentResponse.text(),
        source: "builtin"
      };
    })
  );
}

function loadLocalPosts(builtInIds = new Set()) {
  const saved = localStorage.getItem(storageKey);
  if (!saved) {
    return [];
  }

  try {
    const parsed = JSON.parse(saved);
    if (!Array.isArray(parsed) || !parsed.length) {
      return [];
    }

    return parsed.filter((post) => !builtInIds.has(post.id));
  } catch {
    return [];
  }
}

function saveLocalPosts() {
  localStorage.setItem(
    storageKey,
    JSON.stringify(posts.filter((post) => post.source !== "builtin"))
  );
}

async function initializePosts() {
  let builtInPosts = [];

  try {
    builtInPosts = await loadBuiltInPosts();
  } catch (error) {
    readerTitle.textContent = "无法加载 Markdown 笔记";
    readerExcerpt.textContent = "请通过本地静态服务器访问页面，例如 python3 -m http.server 8080 --bind 127.0.0.1。";
    readerContent.innerHTML = `<p>${escapeHTML(error.message)}</p>`;
  }

  const builtInIds = new Set(builtInPosts.map((post) => post.id));
  const localPosts = loadLocalPosts(builtInIds);
  posts = [...localPosts, ...builtInPosts];
  activePostId = posts[0]?.id;
  syncReaderRoute();
  render();
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
      openPost(post.id);
    });
    postList.append(button);
  });
}

function renderReader() {
  const post = posts.find((item) => item.id === activePostId) || filteredPosts()[0] || posts[0];
  if (!post) {
    readerCategory.textContent = "";
    readerDate.textContent = "";
    readerTitle.textContent = "暂无笔记";
    readerExcerpt.textContent = "还没有加载到可展示的笔记。";
    readerTags.innerHTML = "";
    readerContent.innerHTML = "";
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

function renderDrawer() {
  const query = drawerSearchInput.value.trim().toLowerCase();
  drawerCategoryList.innerHTML = categories()
    .map((category) => `<button type="button" data-drawer-category="${escapeHTML(category)}">${escapeHTML(category)}</button>`)
    .join("");

  drawerTagList.innerHTML = [...new Set(posts.flatMap((post) => post.tags))]
    .map((tag) => `<button type="button" data-drawer-tag="${escapeHTML(tag)}">${escapeHTML(tag)}</button>`)
    .join("");

  const visiblePosts = posts.filter((post) => {
    const text = `${post.title} ${post.category} ${post.tags.join(" ")} ${post.excerpt} ${post.content}`.toLowerCase();
    return text.includes(query);
  });

  drawerPostList.innerHTML = visiblePosts
    .map(
      (post) => `
        <button type="button" class="${post.id === activePostId ? "active" : ""}" data-drawer-post="${escapeHTML(post.id)}">
          <strong>${escapeHTML(post.title)}</strong>
          <span>${escapeHTML(post.category)} · ${formatDate(post.date)}</span>
        </button>
      `
    )
    .join("");
}

function render() {
  renderHomeSummary();
  renderBottomBoard();
  renderCategories();
  renderTags();
  renderPostList();
  renderReader();
  renderDrawer();
}

function openPost(id) {
  activePostId = id;
  render();
  readerPage.classList.remove("is-hidden");
  window.location.hash = `post-${id}`;
  readerPage.scrollIntoView({ behavior: "smooth" });
}

function closeReader() {
  readerPage.classList.add("is-hidden");
  if (window.location.hash.startsWith("#post-")) {
    history.pushState("", document.title, window.location.pathname + window.location.search);
  }
  document.querySelector("#library").scrollIntoView({ behavior: "smooth" });
}

function syncReaderRoute() {
  const hash = decodeURIComponent(window.location.hash);
  if (!hash.startsWith("#post-")) {
    return;
  }

  const id = hash.replace("#post-", "");
  if (posts.some((post) => post.id === id)) {
    activePostId = id;
    readerPage.classList.remove("is-hidden");
  }
}

function openEditor() {
  editorSection.classList.remove("is-hidden");
  editorSection.scrollIntoView({ behavior: "smooth" });
}

function closeEditor() {
  editorSection.classList.add("is-hidden");
}

function openSideNav() {
  renderDrawer();
  sideDrawer.classList.add("is-open");
  sideDrawer.setAttribute("aria-hidden", "false");
}

function closeSideNav() {
  sideDrawer.classList.remove("is-open");
  sideDrawer.setAttribute("aria-hidden", "true");
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

    const imageMatch = line.match(/^!\[([^\]]*)\]\(([^)]+)\)$/);
    if (imageMatch) {
      closeList();
      html.push(renderImage(imageMatch[1], imageMatch[2]));
      return;
    }

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

function renderImage(alt, src) {
  const safeAlt = escapeHTML(alt);
  const safeSrc = escapeAttribute(resolvePostAssetPath(src.trim()));
  const caption = safeAlt ? `<figcaption>${safeAlt}</figcaption>` : "";
  return `<figure><img src="${safeSrc}" alt="${safeAlt}" loading="lazy" />${caption}</figure>`;
}

function resolvePostAssetPath(src) {
  if (/^(https?:)?\/\//.test(src) || src.startsWith("/") || src.startsWith("data:")) {
    return src;
  }

  return `posts/${src.replace(/^\.?\//, "")}`;
}

function escapeHTML(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function escapeAttribute(value) {
  return escapeHTML(value).replaceAll("`", "&#096;");
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

drawerSearchInput.addEventListener("input", renderDrawer);

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

document.querySelectorAll("[data-open-side-nav]").forEach((trigger) => {
  trigger.addEventListener("click", openSideNav);
});

document.querySelectorAll("[data-close-side-nav]").forEach((trigger) => {
  trigger.addEventListener("click", closeSideNav);
});

document.querySelectorAll("[data-close-reader]").forEach((trigger) => {
  trigger.addEventListener("click", closeReader);
});

sideDrawer.addEventListener("click", (event) => {
  const categoryButton = event.target.closest("[data-drawer-category]");
  const tagButton = event.target.closest("[data-drawer-tag]");
  const postButton = event.target.closest("[data-drawer-post]");

  if (categoryButton) {
    activeCategory = categoryButton.dataset.drawerCategory;
    activeTag = "全部";
    closeSideNav();
    render();
    document.querySelector("#library").scrollIntoView({ behavior: "smooth" });
  }

  if (tagButton) {
    const tag = tagButton.dataset.drawerTag;
    const post = posts.find((item) => item.tags.includes(tag));
    activeCategory = post?.category || "全部";
    activeTag = tag;
    closeSideNav();
    render();
    document.querySelector("#library").scrollIntoView({ behavior: "smooth" });
  }

  if (postButton) {
    closeSideNav();
    openPost(postButton.dataset.drawerPost);
  }
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
  readerPage.classList.add("is-hidden");
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
  openPost(post.id);
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
  readerPage.classList.add("is-hidden");
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
    date: new Date().toISOString().slice(0, 10),
    source: "local"
  };

  posts = [post, ...posts];
  activeCategory = "全部";
  activeTag = "全部";
  activePostId = post.id;
  saveLocalPosts();
  postForm.reset();
  render();
  closeEditor();
  openPost(post.id);
});

window.addEventListener("hashchange", syncReaderRoute);

initializePosts();
