# SWIFT — marketing website

Single-page marketing site for SWIFT (lightweight image super-resolution).
Built with **Vite + React 19 + TypeScript + Tailwind v4**, JetBrains Mono throughout,
with hand-authored inline SVG architecture diagrams (no images/diagram libraries).
Supports light and dark themes. Deployed to **Cloudflare Pages**.

Live: https://swift-website.pages.dev

## Develop

```bash
npm install
npm run dev      # vite dev server
npm run build    # type-check + production build to dist/
npm run preview  # serve the built dist/
```

## Deploy (Cloudflare Pages)

```bash
npm run deploy   # builds and runs: wrangler pages deploy dist --project-name=swift-website
```

## Structure

- `src/App.tsx` — the entire site: sections + theme-aware inline SVG diagrams.
- `src/index.css` — Tailwind v4 theme tokens, fonts, grid/glow textures.
- `public/figures/` — paper figures + grayscale LR/SR demo images.
- `public/swift-paper.pdf` — the published paper (opens in-browser from "Read the paper").
