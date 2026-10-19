import type { APIRoute } from "astro";

// The site is one page: list it, dated by the build (the site deploys only when blogpost/ changes).
// Submit https://illuin-tech.github.io/daedalus/sitemap.xml in Google Search Console.
export const GET: APIRoute = ({ site }) => {
  const loc = new URL(import.meta.env.BASE_URL.replace(/\/?$/, "/"), site).href;
  const lastmod = new Date().toISOString().slice(0, 10);
  const xml = `<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>${loc}</loc><lastmod>${lastmod}</lastmod></url>
</urlset>
`;
  return new Response(xml, { headers: { "Content-Type": "application/xml" } });
};
