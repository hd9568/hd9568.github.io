import { getCollection, type CollectionEntry } from 'astro:content';

export const PREVIEW_SIZE = 5;
export const PAGE_SIZE = 10;
export const withBase = (path = '') => `${import.meta.env.BASE_URL.replace(/\/$/, '')}/${path.replace(/^\//, '')}`;
export const postUrl = (id: string) => withBase(`blog/${id}/`);
export const categoryUrl = (slug: string, page = 1) => withBase(`topics/${slug}/${page === 1 ? '' : `${page}/`}`);

const topics = [
  { name: 'C++', slug: 'cpp', label: 'LANGUAGE', symbol: '{ }', description: '从内存与对象模型，到现代 C++ 的工程实践。' },
  { name: '数据结构和算法', slug: 'algorithms', label: 'ALGORITHMS', symbol: '↳', description: '理解数据组织方式，建立分析与解决问题的基础。' },
  { name: '操作系统与多线程', slug: 'systems', label: 'SYSTEMS', symbol: '≋', description: '进程、线程、并发与系统资源管理。' },
  { name: 'CUDA', slug: 'cuda', label: 'GPU COMPUTING', symbol: '▦', description: '走近 GPU 架构，从访存优化到高性能算子。' },
  { name: '搜广推模型', slug: 'search-ads-rec', label: 'SEARCH · ADS · RECOMMENDATION', symbol: '⊙', description: '搜索、广告与推荐中的特征、召回、排序和模型训练推理。' },
  { name: '推理优化', slug: 'inference', label: 'INFERENCE', symbol: '↗', description: 'Attention、KV Cache 与推理框架的实现细节。' },
  { name: '分布式训练', slug: 'training', label: 'DISTRIBUTED TRAINING', symbol: '⋈', description: '从并行策略到通信优化，理解大规模训练。' },
  { name: '论文阅读', slug: 'papers', label: 'PAPER READING', symbol: '▤', description: '论文精读与研究综述，梳理方法、实验和结论边界。' },
  { name: '工程实现', slug: 'engineering', label: 'ENGINEERING', symbol: '⌘', description: '系统设计、算子实现与框架使用，连接原理和工程实践。' },
  { name: '源码学习', slug: 'source', label: 'SOURCE CODE', symbol: '{ }', description: '沿开源项目的调用链，理解模块组织与关键实现。' },
];

export function topicFor(name = '未分类') {
  return topics.find((topic) => topic.name === name) ?? {
    name, slug: `other-${Array.from(name).map((character) => character.codePointAt(0)!.toString(16)).join('-')}`, label: 'NOTES', symbol: '↳', description: '持续整理的技术学习笔记。',
  };
}

export function sortPosts(posts: CollectionEntry<'blog'>[]) {
  return [...posts].sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf() || a.id.localeCompare(b.id));
}

export async function getBlog() {
  const posts = sortPosts(await getCollection('blog'));
  const names = [...new Set(posts.map((post) => post.data.category ?? '未分类'))];
  const orderedNames = [...topics.map((topic) => topic.name).filter((name) => names.includes(name)), ...names.filter((name) => !topics.some((topic) => topic.name === name)).sort()];
  return {
    posts,
    groups: orderedNames.map((name) => ({ ...topicFor(name), posts: posts.filter((post) => (post.data.category ?? '未分类') === name) })),
  };
}

export function readingMinutes(body = '') {
  const text = body.replace(/```[\s\S]*?```/g, '').replace(/<[^>]+>/g, '');
  const chinese = (text.match(/[\u3400-\u9fff]/g) ?? []).length;
  const words = (text.match(/[a-zA-Z0-9]+/g) ?? []).length;
  return Math.max(1, Math.ceil(chinese / 400 + words / 220));
}
