import { defineConfig } from "astro/config";

// Served by GitHub Pages at https://illuin-tech.github.io/daedalus/ (see
// .github/workflows/blog.yml), so every page and asset lives under /daedalus/.
export default defineConfig({
  output: "static",
  site: "https://illuin-tech.github.io",
  base: "/daedalus",
});
