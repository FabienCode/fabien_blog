# Fabien's Notes

一个综合类型的个人记录网站，用来写论文阅读总结、LeetCode 题解、Python 技术笔记、计算机技术博客、读书感悟和生活复盘。

当前版本是纯静态实现，视觉风格偏 Apple 式留白、简洁排版和柔和面板。写作区支持接近 Markdown 的语法，包括标题、列表、引用、代码块、行内代码和链接。

## 本地预览

当前本地服务地址：

```text
http://127.0.0.1:8080
```

由于文章内容从 `posts/` 目录中的 Markdown 文件加载，本地预览需要使用静态服务器，不建议直接双击打开 `index.html`。

如需重新启动本地静态服务器：

```bash
python3 -m http.server 8080 --bind 127.0.0.1
```

## 内容与数据

内置文章保存在 `posts/` 目录中，每篇笔记是一个独立 Markdown 文件，元数据记录在 `posts/index.json`。

通过网页“写笔记”新增的文章会保存在当前浏览器的 `localStorage` 中，适合作为临时单机记录。换浏览器或换设备后不会自动同步。需要长期保存的文章建议写入 `posts/` 并提交到 Git。

默认分类包括：

- 论文
- LeetCode
- Python
- 计算机技术
- 读书感悟
- 生活记录

## 公网访问

推荐部署到以下任一平台：

- GitHub Pages：适合公开仓库，免费稳定。官方文档：https://docs.github.com/en/pages
- Netlify：拖拽整个项目目录即可发布，也支持绑定域名。官方文档：https://docs.netlify.com/site-deploys/create-deploys/
- Vercel：导入 Git 仓库或使用 CLI 部署静态站点。官方文档：https://vercel.com/docs/deployments/deployment-methods

如果只是临时让别人访问本机页面，可以使用 Cloudflare Tunnel 把本地 `8080` 端口暴露到公网。官方文档：https://developers.cloudflare.com/tunnel/

## 后续升级方向

- 增加文章详情页路由，支持分享单篇文章链接。
- 接入 Supabase、Notion 或 GitHub 仓库作为内容后台，实现多设备同步。
