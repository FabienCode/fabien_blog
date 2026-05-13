const postsIndexPath = "posts/index.json";
const storageKey = "personal-notes-posts";
const viewCountsKey = "personal-notes-post-view-counts";
const collectionLimit = 6;
const retiredPostIds = new Set([
  "rag-paper-note",
  "bevfusion-paper-note",
  "leetcode-dp-template",
  "python-context-manager",
  "reading-note"
]);
let posts = [];
let viewCounts = {};
let activeCategory = "全部";
let activeTag = "全部";
let activePostId;
let activeCollectionMode = "recent";
let randomSeed = Date.now();

const categoryFilter = document.querySelector("#categoryFilter");
const tagFilterGroup = document.querySelector("#tagFilterGroup");
const tagFilter = document.querySelector("#tagFilter");
const postList = document.querySelector("#postList");
const searchInput = document.querySelector("#searchInput");
const collectionModeButtons = document.querySelectorAll("[data-collection-mode]");
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

    const localPosts = parsed.filter((post) => !builtInIds.has(post.id) && !retiredPostIds.has(post.id));
    if (localPosts.length !== parsed.length) {
      localStorage.setItem(storageKey, JSON.stringify(localPosts));
    }
    return localPosts;
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

function loadViewCounts() {
  const saved = localStorage.getItem(viewCountsKey);
  if (!saved) {
    return {};
  }

  try {
    const parsed = JSON.parse(saved);
    return parsed && typeof parsed === "object" && !Array.isArray(parsed) ? parsed : {};
  } catch {
    return {};
  }
}

function saveViewCounts() {
  localStorage.setItem(viewCountsKey, JSON.stringify(viewCounts));
}

function recordPostView(id) {
  if (!id) {
    return;
  }

  viewCounts[id] = (Number(viewCounts[id]) || 0) + 1;
  saveViewCounts();
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
  viewCounts = loadViewCounts();
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

function displayedPosts() {
  const visiblePosts = filteredPosts();
  let sortedPosts = [...visiblePosts];

  if (activeCollectionMode === "popular") {
    sortedPosts.sort((a, b) => {
      const viewDiff = (Number(viewCounts[b.id]) || 0) - (Number(viewCounts[a.id]) || 0);
      if (viewDiff) {
        return viewDiff;
      }

      return new Date(b.date) - new Date(a.date);
    });
  } else if (activeCollectionMode === "random") {
    sortedPosts.sort((a, b) => seededRank(a.id) - seededRank(b.id));
  } else {
    sortedPosts.sort((a, b) => new Date(b.date) - new Date(a.date));
  }

  return sortedPosts.slice(0, collectionLimit);
}

function renderCollectionModes() {
  collectionModeButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.collectionMode === activeCollectionMode);
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
  const visiblePosts = displayedPosts();
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
      <span class="post-card-views">阅读 ${Number(viewCounts[post.id]) || 0} 次</span>
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
  const openingSummary = extractOpeningSummary(post.content);
  const bodyContent = openingSummary ? openingSummary.body : post.content;
  readerContent.innerHTML = renderPostBrief(post, openingSummary) + markdownToHTML(bodyContent);
}

function renderPostBrief(post, openingSummary = null) {
  if (openingSummary) {
    return renderOpeningSummaryBrief(openingSummary);
  }

  const headings = extractMarkdownHeadings(post.content)
    .filter((heading) => heading.level === 2)
    .slice(0, 4);
  const keywords = post.tags.slice(0, 6);
  const titleWords = post.title.split(/[：:]/);
  const shortTitle = titleWords.at(-1)?.trim() || post.title;

  return `
    <section class="reader-brief" aria-label="文章速览">
      <div class="brief-head">
        <span>${escapeHTML(post.category)}</span>
        <strong>${escapeHTML(shortTitle)}</strong>
      </div>
      <p>${parseInline(post.excerpt)}</p>
      <div class="brief-grid">
        <article>
          <h3>核心摘要</h3>
          <p>${parseInline(post.excerpt)}</p>
          <span>${escapeHTML(post.category)} · ${formatDate(post.date)}</span>
        </article>
        <article>
          <h3>阅读重点</h3>
          <ul>
            ${
              headings.length
                ? headings.map((heading) => `<li>${parseInline(heading.text)}</li>`).join("")
                : "<li>背景与动机</li><li>方法与实验</li><li>局限与启发</li>"
            }
          </ul>
        </article>
        <article class="brief-wide">
          <h3>关键词</h3>
          <div class="brief-tags">
            ${keywords.map((tag) => `<span>${escapeHTML(tag)}</span>`).join("")}
          </div>
        </article>
      </div>
    </section>
  `;
}

function extractOpeningSummary(markdown) {
  const lines = markdown.replace(/\r\n/g, "\n").split("\n");
  const start = lines.findIndex((line) => /^##\s+.*核心洞察与挑战/.test(line.trim()));
  if (start === -1) {
    return null;
  }

  const intro = lines.findIndex((line, index) => index > start && /^##\s+.*引言/.test(line.trim()));
  const end = intro === -1 ? lines.length : intro;
  const summaryLines = lines.slice(start, end);
  const body = [...lines.slice(0, start), ...lines.slice(end)].join("\n").trim();
  const sections = {
    coreProblem: [],
    priorLimits: [],
    motivation: [],
    method: []
  };
  let activeSection = "";

  summaryLines.forEach((line) => {
    const trimmed = line.trim();
    if (/^##\s+.*核心洞察与挑战/.test(trimmed)) {
      return;
    }
    if (/^###\s+核心问题/.test(trimmed)) {
      activeSection = "coreProblem";
      return;
    }
    if (/^###\s+传统方案局限/.test(trimmed)) {
      activeSection = "priorLimits";
      return;
    }
    if (/^##\s+研究动机/.test(trimmed)) {
      activeSection = "motivation";
      return;
    }
    if (/^##\s+方法论/.test(trimmed)) {
      activeSection = "method";
      return;
    }
    if (activeSection) {
      sections[activeSection].push(line);
    }
  });

  if (!Object.values(sections).some((sectionLines) => sectionLines.join("").trim())) {
    return null;
  }

  return { sections, body };
}

function renderOpeningSummaryBrief(openingSummary) {
  const { sections } = openingSummary;
  const renderSection = (sectionLines) => markdownToHTML(sectionLines.join("\n").trim());

  return `
    <section class="reader-brief reader-brief-summary" aria-label="论文开篇总结">
      <div class="brief-head">
        <span>Paper Brief</span>
        <strong>核心洞察与挑战</strong>
      </div>
      <div class="brief-grid brief-summary-grid">
        <article>
          <h3>核心问题</h3>
          ${renderSection(sections.coreProblem)}
        </article>
        <article>
          <h3>传统方案局限</h3>
          ${renderSection(sections.priorLimits)}
        </article>
        <article class="brief-wide">
          <h3>研究动机</h3>
          ${renderSection(sections.motivation)}
        </article>
        <article class="brief-wide brief-method">
          <h3>方法论（主要模块简介）</h3>
          ${renderSection(sections.method)}
        </article>
      </div>
    </section>
  `;
}

function extractMarkdownHeadings(markdown) {
  return markdown
    .replace(/\r\n/g, "\n")
    .split("\n")
    .map((line) => line.trim().match(/^(#{1,6})\s+(.+)$/))
    .filter(Boolean)
    .map((match) => ({
      level: match[1].length,
      text: match[2].replace(/\s+#+$/, "")
    }));
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
  renderCollectionModes();
  renderCategories();
  renderTags();
  renderPostList();
  renderReader();
  renderDrawer();
}

function setReadingMode(isReading) {
  document.body.classList.toggle("is-reading-post", isReading);
}

function openPost(id) {
  activePostId = id;
  recordPostView(id);
  render();
  readerPage.classList.remove("is-hidden");
  editorSection.classList.add("is-hidden");
  setReadingMode(true);
  window.location.hash = `post-${id}`;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function closeReader() {
  readerPage.classList.add("is-hidden");
  setReadingMode(false);
  if (window.location.hash.startsWith("#post-")) {
    history.pushState("", document.title, window.location.pathname + window.location.search);
  }
  document.querySelector("#library").scrollIntoView({ behavior: "smooth" });
}

function syncReaderRoute() {
  const hash = decodeURIComponent(window.location.hash);
  if (!hash.startsWith("#post-")) {
    readerPage.classList.add("is-hidden");
    setReadingMode(false);
    return;
  }

  const id = hash.replace("#post-", "");
  if (posts.some((post) => post.id === id)) {
    activePostId = id;
    renderReader();
    readerPage.classList.remove("is-hidden");
    editorSection.classList.add("is-hidden");
    setReadingMode(true);
  }
}

function openEditor() {
  readerPage.classList.add("is-hidden");
  setReadingMode(false);
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
  let codeLanguage = "";
  let codeLines = [];
  let listType = "";
  let paragraphLines = [];

  function closeParagraph() {
    if (paragraphLines.length) {
      html.push(`<p>${parseInline(paragraphLines.join(" "))}</p>`);
      paragraphLines = [];
    }
  }

  function closeList() {
    if (listType) {
      html.push(`</${listType}>`);
      listType = "";
    }
  }

  function closeOpenBlocks() {
    closeParagraph();
    closeList();
  }

  function renderCodeBlock() {
    const languageClass = codeLanguage ? ` class="language-${escapeAttribute(codeLanguage)}"` : "";
    const languageLabel = escapeAttribute(codeLanguage || "code");
    html.push(
      `<div class="code-card" data-language="${languageLabel}"><pre><code${languageClass}>${escapeHTML(codeLines.join("\n"))}</code></pre></div>`
    );
    codeLines = [];
    codeLanguage = "";
  }

  function isTableStart(index) {
    const current = lines[index]?.trim();
    const next = lines[index + 1]?.trim();
    return Boolean(
      current &&
        next &&
        current.includes("|") &&
        /^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?$/.test(next)
    );
  }

  function splitTableRow(row) {
    return row
      .trim()
      .replace(/^\|/, "")
      .replace(/\|$/, "")
      .split("|")
      .map((cell) => cell.trim());
  }

  function renderTable(startIndex) {
    const headers = splitTableRow(lines[startIndex]);
    const alignments = splitTableRow(lines[startIndex + 1]).map((cell) => {
      if (/^:-+:$/.test(cell)) return "center";
      if (/-+:$/.test(cell)) return "right";
      if (/^:-+/.test(cell)) return "left";
      return "";
    });
    const rows = [];
    let index = startIndex + 2;

    while (index < lines.length && lines[index].trim().includes("|")) {
      rows.push(splitTableRow(lines[index]));
      index += 1;
    }

    const headerHTML = headers
      .map((header, cellIndex) => {
        const align = alignments[cellIndex] ? ` style="text-align:${alignments[cellIndex]}"` : "";
        return `<th${align}>${parseInline(header)}</th>`;
      })
      .join("");
    const rowsHTML = rows
      .map(
        (row) =>
          `<tr>${headers
            .map((_, cellIndex) => {
              const align = alignments[cellIndex] ? ` style="text-align:${alignments[cellIndex]}"` : "";
              return `<td${align}>${parseInline(row[cellIndex] || "")}</td>`;
            })
            .join("")}</tr>`
      )
      .join("");

    html.push(`<div class="table-scroll"><table><thead><tr>${headerHTML}</tr></thead><tbody>${rowsHTML}</tbody></table></div>`);
    return index;
  }

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();
    const fenceMatch = trimmed.match(/^```([A-Za-z0-9_+#.-]*)/);
    if (fenceMatch) {
      closeOpenBlocks();
      if (inCode) {
        renderCodeBlock();
        inCode = false;
      } else {
        inCode = true;
        codeLanguage = fenceMatch[1] || "";
      }
      continue;
    }

    if (inCode) {
      codeLines.push(line);
      continue;
    }

    if (!trimmed) {
      closeOpenBlocks();
      continue;
    }

    if (/^[-*_]{3,}$/.test(trimmed)) {
      closeOpenBlocks();
      html.push("<hr />");
      continue;
    }

    const imageMatch = trimmed.match(/^!\[([^\]]*)\]\(([^)\s]+)(?:\s+"([^"]+)")?\)$/);
    if (imageMatch) {
      closeOpenBlocks();
      html.push(renderImage(imageMatch[1], imageMatch[2], imageMatch[3]));
      continue;
    }

    if (isTableStart(index)) {
      closeOpenBlocks();
      index = renderTable(index) - 1;
      continue;
    }

    if (/^>\s?/.test(line)) {
      closeOpenBlocks();
      const quoteLines = [];
      while (index < lines.length && /^>\s?/.test(lines[index])) {
        quoteLines.push(lines[index].replace(/^>\s?/, ""));
        index += 1;
      }
      index -= 1;
      html.push(`<blockquote>${markdownToHTML(quoteLines.join("\n"))}</blockquote>`);
      continue;
    }

    const headingMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (headingMatch) {
      const level = Math.min(6, Math.max(1, headingMatch[1].length));
      closeOpenBlocks();
      html.push(`<h${level}>${parseInline(headingMatch[2].replace(/\s+#+$/, ""))}</h${level}>`);
      continue;
    }

    const unorderedMatch = line.match(/^\s*[-*+]\s+(.+)$/);
    const orderedMatch = line.match(/^\s*\d+[.)]\s+(.+)$/);
    if (unorderedMatch || orderedMatch) {
      closeParagraph();
      const nextListType = unorderedMatch ? "ul" : "ol";
      if (listType && listType !== nextListType) {
        closeList();
      }
      if (!listType) {
        html.push(`<${nextListType}>`);
        listType = nextListType;
      }
      html.push(`<li>${parseInline((unorderedMatch || orderedMatch)[1])}</li>`);
      continue;
    }

    closeList();
    paragraphLines.push(trimmed);
  }

  closeOpenBlocks();

  if (inCode) {
    renderCodeBlock();
  }

  return html.join("");
}

function parseInline(value) {
  const codeSpans = [];
  let text = String(value).replace(/`([^`]+)`/g, (_, code) => {
    const placeholder = `\u0000CODE${codeSpans.length}\u0000`;
    codeSpans.push(`<code>${escapeHTML(code)}</code>`);
    return placeholder;
  });

  text = escapeHTML(text)
    .replace(/\*\*([\s\S]+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__([\s\S]+?)__/g, "<strong>$1</strong>")
    .replace(/~~([\s\S]+?)~~/g, "<del>$1</del>")
    .replace(/(^|[^\*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+&quot;[^&]+&quot;)?\)/g, (_, label, href) => {
      const safeHref = sanitizeHref(href);
      const externalAttrs = /^(https?:)?\/\//.test(safeHref) ? ' target="_blank" rel="noreferrer"' : "";
      return `<a href="${escapeAttribute(safeHref)}"${externalAttrs}>${label}</a>`;
    });

  codeSpans.forEach((code, index) => {
    text = text.replace(`\u0000CODE${index}\u0000`, code);
  });

  return text;
}

function renderImage(alt, src, title = "") {
  const safeAlt = escapeHTML(alt);
  const safeSrc = escapeAttribute(resolvePostAssetPath(src.trim()));
  const captionText = title || alt;
  const caption = captionText ? `<figcaption>${parseInline(captionText)}</figcaption>` : "";
  return `<figure><img src="${safeSrc}" alt="${safeAlt}" loading="lazy" />${caption}</figure>`;
}

function resolvePostAssetPath(src) {
  if (/^(https?:)?\/\//.test(src) || src.startsWith("/") || src.startsWith("data:")) {
    return src;
  }

  return `posts/${src.replace(/^\.?\//, "")}`;
}

function sanitizeHref(href) {
  const value = String(href || "").trim();
  if (/^(https?:|mailto:|#|\/)/i.test(value)) {
    return value;
  }

  if (/^(javascript:|data:)/i.test(value)) {
    return "#";
  }

  return value;
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

function seededRank(value) {
  const text = `${value}-${randomSeed}`;
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) {
    hash = (hash << 5) - hash + text.charCodeAt(index);
    hash |= 0;
  }

  return Math.abs(hash);
}

searchInput.addEventListener("input", () => {
  const first = filteredPosts()[0];
  activePostId = first?.id || activePostId;
  render();
});

collectionModeButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activeCollectionMode = button.dataset.collectionMode;
    if (activeCollectionMode === "random") {
      randomSeed = Date.now();
    }

    const first = displayedPosts()[0];
    activePostId = first?.id || activePostId;
    render();
  });
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
    readerPage.classList.add("is-hidden");
    setReadingMode(false);
    document.querySelector("#library").scrollIntoView({ behavior: "smooth" });
  }

  if (tagButton) {
    const tag = tagButton.dataset.drawerTag;
    const post = posts.find((item) => item.tags.includes(tag));
    activeCategory = post?.category || "全部";
    activeTag = tag;
    closeSideNav();
    render();
    readerPage.classList.add("is-hidden");
    setReadingMode(false);
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
  setReadingMode(false);
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
  setReadingMode(false);
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
