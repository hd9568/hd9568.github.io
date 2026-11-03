import { getCollection } from 'astro:content';
import rss from '@astrojs/rss';
import { SITE_DESCRIPTION, SITE_TITLE } from '../consts';
import { postUrl, sortPosts } from '../lib/blog';

export async function GET(context) {
  const posts = await getCollection('blog');
  return rss({
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    site: context.site,
    items: sortPosts(posts).map((post) => ({
      ...post.data,
      link: postUrl(post.id),
    })),
  });
}
