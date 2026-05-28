# Blog Workflow

## 1. 项目目标

- 从 0 搭建一个适合长期维护的个人博客。
- 使用 GitHub Pages 作为托管平台。
- 博客内容聚焦 AI Infra、训练优化、推理优化岗位相关知识。
- 第一篇文章围绕 `infra_list.md` 的第一个知识点展开。

## 2. 技术选型

- 静态站点框架：Astro
- 内容组织：Markdown
- 自动部署：GitHub Actions + GitHub Pages
- 当前对外名称：霍墨墨

选择这个方案的原因：

- 页面足够简洁，后续美化成本低。
- 写文章直接用 Markdown，适合持续输出学习笔记。
- GitHub Pages 托管稳定，适合个人技术博客。
- Astro 默认产出静态文件，部署轻量。

## 3. 环境检查

在本机现有环境中检测到：

- `node -v` -> `v24.16.0`
- `npm -v` -> `11.13.0`
- `git --version` -> `2.50.1`
- `python3 --version` -> `3.9.6`

本次搭建没有额外安装系统级依赖，现有 Node.js 环境已满足要求。

## 4. 初始化过程

### 4.1 使用官方模板生成博客骨架

执行命令：

```bash
npm create astro@latest temp-blog -- --template blog --install --git=false --yes
```

说明：

- `create-astro` 会通过 `npx` 临时拉起脚手架。
- 模板使用的是 Astro 官方 blog starter。
- 生成目录为 `temp-blog`，方便先检查模板结构再迁移到当前目录。

### 4.2 合并到当前目录

执行命令：

```bash
rsync -a --exclude node_modules --exclude .git /Users/bytedance/Desktop/blog/temp-blog/ /Users/bytedance/Desktop/blog/
```

这样可以：

- 保留当前目录已有的 `infra_list.md`
- 把博客源码复制到当前项目根目录
- 不复制模板目录中的 `.git` 和 `node_modules`

## 5. 本次定制内容

### 5.1 站点信息

- 站点标题：`霍墨墨 | AI Infra Notes`
- 对外名称：霍墨墨
- GitHub 用户名：`hd9568`
- 邮箱：`2358440725@qq.com`

### 5.2 页面结构

已配置页面：

- 首页：展示个人定位、技术方向、最新文章
- 文章列表页：集中查看博客文章
- 关于页：介绍背景、求职方向和博客目标
- 文章详情页：统一展示标题、摘要、正文与更新时间

### 5.3 样式设计

本次界面风格定位：

- 简洁
- 明亮
- 适合技术博客阅读
- 兼顾个人主页展示感

## 6. 自动部署配置

已新增 GitHub Actions 工作流文件：

- `.github/workflows/deploy.yml`

工作流行为：

1. 当 `main` 分支有新的 push 时自动触发。
2. 安装依赖并执行 `npm run build`。
3. 将 `dist` 目录部署到 GitHub Pages。

## 7. 仓库发布建议

推荐把 GitHub 仓库命名为：

```text
hd9568.github.io
```

这样部署后的访问地址最干净：

```text
https://hd9568.github.io
```

如果你最终使用的仓库名不是 `hd9568.github.io`，当前配置也会根据 GitHub 仓库名自动设置 `base` 路径，以适配项目页部署。

## 8. 本地开发命令

首次在当前目录安装依赖：

```bash
npm install
```

启动本地开发环境：

```bash
npm run dev
```

构建生产版本：

```bash
npm run build
```

本地预览构建结果：

```bash
npm run preview
```

## 9. GitHub Pages 启用步骤

当你把项目推到 GitHub 后，建议检查以下配置：

1. 进入仓库 `Settings`。
2. 打开 `Pages`。
3. 确认 `Source` 使用 GitHub Actions。
4. 推送到 `main` 分支后等待 Actions 执行完成。

## 10. 内容维护方式

后续新增文章时，直接在：

```text
src/content/blog/
```

目录下新增 Markdown 文件即可。

建议每篇文章都保持以下结构：

1. 标题
2. 内容简介
3. 核心概念解释
4. 简单代码示例
5. 常见误区或面试回答方式
6. 总结

## 11. 首篇文章安排

已创建首篇文章：

- `src/content/blog/pointer-reference-basics.md`

主题：

- 指针与引用的底层区别
- 悬垂指针与野指针

写作目标：

- 通俗易懂
- 先讲直觉，再讲底层差异
- 配简单 C++ 示例代码
- 适合作为秋招复习材料
