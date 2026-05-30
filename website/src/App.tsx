import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";

/* ─────────────────────────────────────────────────────────────────────────
   SWIFT — marketing site.
   Single file by design. Only React is imported; everything else is Tailwind
   theme tokens (from index.css) + hand-authored inline SVG diagrams.
   Strictly monochrome — works in both dark (white on black) and light
   (black on white) themes. No decorative color; the only accent is the
   inverted highlight, reserved for SWIFT itself.
   ──────────────────────────────────────────────────────────────────────── */

const REPO = "https://github.com/Abhishek-Aditya-bs/SWIFT";
const PAPER_PDF = "/swift-paper.pdf";
const PAPER_URL = "https://ojs.bonviewpress.com/index.php/AIA/article/view/1930";
const AUTHORS = [
  ["Vishal Ramesha", "https://github.com/iVishalr", "Vishal"],
  ["Abhishek Aditya BS", "https://github.com/Abhishek-Aditya-bs", "Abhishek"],
  ["Yashas Kadambi", "https://github.com/Yashas120", "Yashas"],
  ["T Vijay Prashant", "https://github.com/tvijayprashant", "Vijay"],
  ["Shylaja S S", "https://scholar.google.co.in/citations?user=X365OjgAAAAJ&hl=en", "Shylaja"],
] as const;

/* Theme-aware diagram palette. Every SVG colour resolves through usePalette()
   so the hand-drawn diagrams invert cleanly between dark and light. */
type Palette = {
  ink: string;
  sub: string;
  mute: string;
  line: string;
  edge: string;
  panel: string;
  panel2: string;
  bg: string;
  hi: string;
  hiText: string;
};
const DARK: Palette = {
  ink: "#fafafa",
  sub: "#a1a1aa",
  mute: "#7a7a85",
  line: "#27272a",
  edge: "#3a3a40",
  panel: "#0b0b0c",
  panel2: "#161618",
  bg: "#000000",
  hi: "#ffffff",
  hiText: "#0a0a0a",
};
const LIGHT: Palette = {
  ink: "#0a0a0a",
  sub: "#3f3f46",
  mute: "#6b6b75",
  line: "#e4e4e8",
  edge: "#bcbcc6",
  panel: "#ffffff",
  panel2: "#f4f4f5",
  bg: "#ffffff",
  hi: "#0a0a0a",
  hiText: "#ffffff",
};

const ThemeCtx = createContext<{ dark: boolean; toggle: () => void }>({ dark: true, toggle: () => {} });
const useTheme = () => useContext(ThemeCtx);
const usePalette = (): Palette => (useContext(ThemeCtx).dark ? DARK : LIGHT);

export default function App() {
  const [dark, setDark] = useState<boolean>(() => {
    if (typeof document !== "undefined") return document.documentElement.classList.contains("dark");
    return true;
  });
  const toggle = () => {
    setDark((d) => {
      const next = !d;
      const root = document.documentElement;
      root.classList.toggle("dark", next);
      try {
        localStorage.setItem("swift-theme", next ? "dark" : "light");
      } catch {
        /* ignore */
      }
      return next;
    });
  };
  return (
    <ThemeCtx.Provider value={{ dark, toggle }}>
      <div className="min-h-screen bg-background text-foreground antialiased">
        <GlobalFX />
        <DiagramDefs />
        <Nav />
        <main>
          <Hero />
          <Problem />
          <Architecture />
          <Block />
          <Components />
          <Results />
          <Speed />
          <Qualitative />
          <Recipe />
          <Quickstart />
          <Cite />
        </main>
        <Footer />
      </div>
    </ThemeCtx.Provider>
  );
}

/* ───────────────────────────────── effects ─────────────────────────────── */

function GlobalFX() {
  return (
    <style>{`
      @keyframes sw-dash  { to { stroke-dashoffset: -28; } }
      @keyframes sw-pulse { 0%,100% { opacity:.28 } 50% { opacity:.9 } }
      @keyframes sw-spin  { to { transform: rotate(360deg); } }
      @keyframes sw-scan  { 0%,100% { opacity:.15 } 50% { opacity:.6 } }
      .flow      { stroke-dasharray: 5 9;  animation: sw-dash 1s linear infinite; }
      .flow-slow { stroke-dasharray: 4 12; animation: sw-dash 2.3s linear infinite; }
      .pulse     { animation: sw-pulse 2.6s ease-in-out infinite; }
      .spin-slow { animation: sw-spin 16s linear infinite; transform-box: fill-box; transform-origin: center; }
      .scan      { animation: sw-scan 3.2s ease-in-out infinite; }
      @media (prefers-reduced-motion: reduce) {
        .flow,.flow-slow,.pulse,.spin-slow,.scan { animation: none; }
      }
    `}</style>
  );
}

/* Shared SVG defs (markers/filters/gradients) referenced by id from every
   diagram — SVG id refs resolve document-wide. */
function DiagramDefs() {
  const C = usePalette();
  return (
    <svg width="0" height="0" className="absolute" aria-hidden="true">
      <defs>
        <marker id="arr" markerWidth="9" markerHeight="9" refX="6.4" refY="4" orient="auto">
          <path d="M0 0 L8 4 L0 8 z" fill={C.mute} />
        </marker>
        <marker id="arrw" markerWidth="9" markerHeight="9" refX="6.4" refY="4" orient="auto">
          <path d="M0 0 L8 4 L0 8 z" fill={C.ink} />
        </marker>
        <filter id="soft" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation="3.2" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
        {/* Soft drop shadows give the white nodes depth in light theme. */}
        <filter id="nshadow" x="-40%" y="-40%" width="180%" height="180%">
          <feDropShadow dx="0" dy="1.5" stdDeviation="3" floodColor="#1c1c22" floodOpacity="0.13" />
        </filter>
        <filter id="nshadowLg" x="-60%" y="-60%" width="220%" height="220%">
          <feDropShadow dx="0" dy="3" stdDeviation="6" floodColor="#0a0a0a" floodOpacity="0.18" />
        </filter>
      </defs>
    </svg>
  );
}

/* IntersectionObserver fade-up wrapper for a touch of motion on scroll. */
function Reveal({ children, className = "" }: { children: ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [shown, setShown] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setShown(true);
          io.disconnect();
        }
      },
      { threshold: 0.12 },
    );
    io.observe(el);
    return () => io.disconnect();
  }, []);
  return (
    <div
      ref={ref}
      className={`transition-all duration-700 ease-out ${shown ? "translate-y-0 opacity-100" : "translate-y-3 opacity-0"} ${className}`}
    >
      {children}
    </div>
  );
}

/* ───────────────────────────────── chrome ──────────────────────────────── */

function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <div className="inline-flex w-fit max-w-full items-center gap-2 self-start rounded-full border border-border bg-muted/30 px-3 py-1 text-[11px] uppercase tracking-[0.2em] text-muted-foreground">
      {children}
    </div>
  );
}

function Section({ id, className = "", children }: { id?: string; className?: string; children: ReactNode }) {
  return (
    <section id={id} className={`mx-auto w-full max-w-6xl px-5 py-20 md:py-28 ${className}`}>
      {children}
    </section>
  );
}

function SectionHead({ kicker, title, lead }: { kicker: string; title: ReactNode; lead?: ReactNode }) {
  return (
    <div className="flex max-w-3xl flex-col gap-4">
      <Eyebrow>{kicker}</Eyebrow>
      <h2 className="text-3xl font-bold tracking-tight md:text-4xl">{title}</h2>
      {lead && <p className="prose max-w-2xl text-sm leading-relaxed text-muted-foreground md:text-base">{lead}</p>}
    </div>
  );
}

function Logo({ size = 26 }: { size?: number }) {
  const C = usePalette();
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <rect width="32" height="32" rx="7" fill={C.hi} />
      <rect x="6" y="6" width="9" height="9" fill={C.hiText} />
      <rect x="6" y="17" width="9" height="9" fill={C.hiText} opacity="0.5" />
      <g fill={C.hiText}>
        <rect x="18" y="6" width="3.6" height="3.6" />
        <rect x="22.4" y="6" width="3.6" height="3.6" opacity="0.5" />
        <rect x="18" y="10.4" width="3.6" height="3.6" opacity="0.5" />
        <rect x="22.4" y="10.4" width="3.6" height="3.6" />
        <rect x="18" y="17" width="3.6" height="3.6" opacity="0.5" />
        <rect x="22.4" y="17" width="3.6" height="3.6" />
        <rect x="18" y="21.4" width="3.6" height="3.6" />
        <rect x="22.4" y="21.4" width="3.6" height="3.6" opacity="0.5" />
      </g>
    </svg>
  );
}

function ThemeToggle() {
  const { dark, toggle } = useTheme();
  return (
    <button
      onClick={toggle}
      aria-label={dark ? "Switch to light theme" : "Switch to dark theme"}
      title={dark ? "Light theme" : "Dark theme"}
      className="inline-flex h-8 w-8 items-center justify-center rounded-lg border border-border text-muted-foreground transition-colors hover:text-foreground"
    >
      {dark ? (
        // sun
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
          <circle cx="12" cy="12" r="4" />
          <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M19.1 4.9l-1.4 1.4M6.3 17.7l-1.4 1.4" />
        </svg>
      ) : (
        // moon
        <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
          <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />
        </svg>
      )}
    </button>
  );
}

function GitHubMark({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" className={className} aria-hidden="true">
      <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z" />
    </svg>
  );
}

function ArrowR() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" aria-hidden="true" className="inline-block">
      <path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function PaperIcon() {
  return (
    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M6 2h8l4 4v16H6z" />
      <path d="M14 2v4h4M9 13h6M9 17h6M9 9h2" />
    </svg>
  );
}

function Nav() {
  const links = [
    ["Motivation", "#motivation"],
    ["Architecture", "#architecture"],
    ["Components", "#components"],
    ["Results", "#results"],
    ["Paper", "#paper"],
  ];
  return (
    <header className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-14 w-full max-w-6xl items-center justify-between px-5">
        <a href="#top" className="flex items-center gap-2.5">
          <Logo size={24} />
          <span className="text-sm font-semibold tracking-[0.22em]">SWIFT</span>
        </a>
        <nav className="hidden items-center gap-7 md:flex">
          {links.map(([label, href]) => (
            <a key={href} href={href} className="text-[13px] text-muted-foreground transition-colors hover:text-foreground">
              {label}
            </a>
          ))}
        </nav>
        <div className="flex items-center gap-2">
          <ThemeToggle />
          <a
            href={REPO}
            className="inline-flex items-center gap-2 rounded-lg border border-border px-3 py-1.5 text-[13px] text-muted-foreground transition-colors hover:text-foreground"
          >
            <GitHubMark /> <span className="hidden sm:inline">GitHub</span>
          </a>
          <a
            href="#results"
            className="inline-flex items-center gap-1.5 rounded-lg bg-foreground px-3.5 py-1.5 text-[13px] font-medium text-background transition-opacity hover:opacity-90"
          >
            See results
          </a>
        </div>
      </div>
    </header>
  );
}

/* ───────────────────────────── svg primitives ──────────────────────────── */

function Scroller({ min, children }: { min: number; children: ReactNode }) {
  return (
    <div className="mask-fade-x -mx-5 overflow-x-auto px-5 pb-2">
      <div style={{ minWidth: min }}>{children}</div>
    </div>
  );
}

type NodeProps = {
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  sub?: string;
  accent?: "spatial" | "freq" | "none";
  fill?: string;
  emph?: boolean;
};

function Node({ x, y, w, h, title, sub, accent = "none", fill, emph = false }: NodeProps) {
  const C = usePalette();
  const { dark } = useTheme();
  const cx = x + w / 2;
  // Emphasis is theme-aware. Dark: invert to a solid light box (it pops on black).
  // Light: keep a white box but make it the hero with a bold dark border + a
  // stronger drop shadow — never a heavy black slab.
  const emphDark = emph && dark;
  const boxFill = emphDark ? C.hi : fill ?? C.panel;
  const boxStroke = emphDark ? C.hi : emph ? C.ink : C.edge;
  const boxStrokeW = emph && !dark ? 2 : emph ? 1.4 : 1;
  const titleFill = emphDark ? C.hiText : C.ink;
  const subFill = emphDark ? C.hiText : C.sub;
  const filterId = emphDark ? "soft" : !dark ? (emph ? "nshadowLg" : "nshadow") : undefined;
  return (
    <g>
      <rect x={x} y={y} width={w} height={h} rx={9} fill={boxFill} stroke={boxStroke} strokeWidth={boxStrokeW} filter={filterId ? `url(#${filterId})` : undefined} />
      {accent === "spatial" && <rect x={x} y={y + 8} width={3.5} height={h - 16} rx={2} fill={C.ink} />}
      {accent === "freq" && (
        <g fill={C.ink}>
          <rect x={x} y={y + 8} width={3.5} height={(h - 22) / 3} rx={2} />
          <rect x={x} y={y + 8 + (h - 16) / 3 + 2} width={3.5} height={(h - 22) / 3} rx={2} />
          <rect x={x} y={y + 8 + (2 * (h - 16)) / 3 + 4} width={3.5} height={(h - 22) / 3} rx={2} />
        </g>
      )}
      <text x={cx} y={sub ? y + h / 2 - 3 : y + h / 2 + 4} textAnchor="middle" fontSize={13} fontWeight={700} fontFamily="'JetBrains Mono', monospace" fill={titleFill}>
        {title}
      </text>
      {sub && (
        <text x={cx} y={y + h / 2 + 13} textAnchor="middle" fontSize={10} fontFamily="'JetBrains Mono', monospace" fill={subFill} opacity={emphDark ? 0.6 : 0.85}>
          {sub}
        </text>
      )}
    </g>
  );
}

function Chip({ x, y, text, size = 11, color, anchor = "start", weight = 400 }: { x: number; y: number; text: string; size?: number; color?: string; anchor?: "start" | "middle" | "end"; weight?: number }) {
  const C = usePalette();
  const w = text.length * size * 0.62 + 14;
  const rx = anchor === "middle" ? x - w / 2 : anchor === "end" ? x - w : x;
  return (
    <g>
      <rect x={rx} y={y - size + 1} width={w} height={size + 8} rx={4} fill={C.bg} opacity={0.92} />
      <text x={x} y={y + 3} fontSize={size} fontFamily="'JetBrains Mono', monospace" fill={color ?? C.sub} textAnchor={anchor} fontWeight={weight}>
        {text}
      </text>
    </g>
  );
}

function Plus({ x, y, r = 11 }: { x: number; y: number; r?: number }) {
  const C = usePalette();
  return (
    <g>
      <circle cx={x} cy={y} r={r} fill={C.panel} stroke={C.edge} />
      <path d={`M${x - 5} ${y} H${x + 5} M${x} ${y - 5} V${y + 5}`} stroke={C.ink} strokeWidth={1.5} />
    </g>
  );
}

/* base grey wire + animated marching-dash overlay = "data flowing". */
function Flow({ d, slow = false, marker = true, sw = 1.6 }: { d: string; slow?: boolean; marker?: boolean; sw?: number }) {
  const C = usePalette();
  return (
    <g>
      <path d={d} fill="none" stroke={C.mute} strokeWidth={sw} markerEnd={marker ? "url(#arr)" : undefined} />
      <path d={d} fill="none" stroke={C.ink} strokeWidth={sw} className={slow ? "flow-slow" : "flow"} />
    </g>
  );
}

/* A real grayscale image (LR pixelated / SR sharp) clipped to a rounded frame —
   shows what super-resolution actually does, on-theme in black and white. */
function Img({ x, y, size, href, id }: { x: number; y: number; size: number; href: string; id: string }) {
  const C = usePalette();
  return (
    <g>
      <clipPath id={id}>
        <rect x={x} y={y} width={size} height={size} rx={7} />
      </clipPath>
      <image href={href} x={x} y={y} width={size} height={size} preserveAspectRatio="xMidYMid slice" clipPath={`url(#${id})`} />
      <rect x={x} y={y} width={size} height={size} rx={7} fill="none" stroke={C.edge} />
    </g>
  );
}

/* ──────────────────────────────── hero ─────────────────────────────────── */

function Hero() {
  return (
    <div id="top" className="relative overflow-hidden border-b border-border">
      <div className="bg-grid bg-grid-fade pointer-events-none absolute inset-0" />
      <div className="glow pointer-events-none absolute inset-x-0 top-0 h-[460px]" />
      <Section className="relative !py-20 md:!py-24">
        <div className="flex flex-col items-start gap-6">
          <Eyebrow>lightweight · super-resolution · swinV2 + fourier</Eyebrow>
          <h1 className="max-w-4xl text-4xl font-bold leading-[1.06] tracking-tight md:text-6xl">
            Restore the detail.
            <br />
            <span className="text-muted-foreground">Drop the compute.</span>
          </h1>
          <p className="prose max-w-2xl text-base leading-relaxed text-muted-foreground md:text-lg">
            SWIFT is a lightweight single-image super-resolution network that fuses{" "}
            <span className="text-foreground">SwinV2 transformers</span> with{" "}
            <span className="text-foreground">Fast Fourier Convolutions</span>. It matches SwinIR&apos;s
            reconstruction quality while using <span className="text-foreground">34% fewer parameters</span> and running{" "}
            <span className="text-foreground">up to 60% faster</span> at inference.
          </p>
          <div className="flex flex-wrap items-center gap-3 pt-1">
            <a href="#results" className="inline-flex items-center gap-2 rounded-lg bg-foreground px-4 py-2.5 text-sm font-medium text-background transition-opacity hover:opacity-90">
              See the results <ArrowR />
            </a>
            <a href={PAPER_PDF} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 rounded-lg border border-border px-4 py-2.5 text-sm text-muted-foreground transition-colors hover:text-foreground">
              <PaperIcon /> Read the paper
            </a>
            <a href={REPO} className="inline-flex items-center gap-2 rounded-lg border border-border px-4 py-2.5 text-sm text-muted-foreground transition-colors hover:text-foreground">
              <GitHubMark /> View source
            </a>
          </div>
          <p className="pt-1 font-mono text-xs text-muted-foreground/80">
            PyTorch · DIV2K · ×2 / ×3 / ×4 · TorchServe · Docker · MIT
          </p>
        </div>

        <Reveal className="mt-12 md:mt-16">
          <div className="rounded-2xl border border-border bg-card/30 p-4 md:p-6">
            <HeroDiagram />
          </div>
        </Reveal>

        <div className="mt-8 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-border bg-border sm:grid-cols-4">
          {[
            ["596K", "parameters (×4)"],
            ["−34%", "vs SwinIR params"],
            ["≤ −60%", "inference time"],
            ["5", "benchmark sets"],
          ].map(([n, l]) => (
            <div key={l} className="bg-background px-5 py-6">
              <div className="text-2xl font-semibold tracking-tight tabular-nums md:text-3xl">{n}</div>
              <div className="mt-1 text-xs text-muted-foreground">{l}</div>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}

function HeroDiagram() {
  const C = usePalette();
  return (
    <Scroller min={820}>
      <svg viewBox="0 0 1100 300" className="block h-auto w-full" role="img" aria-label="SWIFT super-resolution pipeline: a low-resolution butterfly image passes through shallow features, deep feature extraction, and pixel-shuffle reconstruction to a sharp high-resolution output">
        {/* LR */}
        <Img id="hero-lr" x={26} y={90} size={120} href="/figures/monarch-lr.png" />
        <Chip x={86} y={236} text="LR · low-res" anchor="middle" color={C.sub} />
        <Chip x={86} y={254} text="H × W × 3" anchor="middle" size={10} color={C.mute} />

        {/* shallow */}
        <Flow d="M150 150 H 196" />
        <Node x={198} y={120} w={104} h={60} title="Shallow" sub="3×3 conv" />

        {/* deep container */}
        <rect x={330} y={70} width={392} height={160} rx={12} fill={C.panel} stroke={C.line} strokeDasharray="3 6" />
        {/* residual skip from shallow over the deep block */}
        <path d="M302 134 C 380 30, 660 30, 742 134" fill="none" stroke={C.mute} strokeWidth={1.4} strokeDasharray="2 5" markerEnd="url(#arr)" />
        <Chip x={522} y={46} text="global residual" size={10} color={C.mute} anchor="middle" />
        <Chip x={526} y={94} text="deep feature extraction" size={10} color={C.mute} anchor="middle" />
        <Flow d="M302 150 H 350" />
        {[0, 1, 2, 3].map((i) => (
          <Node key={i} x={350 + i * 70} y={112} w={56} h={76} title="FSTB" sub={`#${i + 1}`} accent="spatial" />
        ))}
        <Node x={636} y={112} w={70} h={76} title="conv" sub="3×3" />
        {[0, 1, 2, 3].map((i) => (
          <Flow key={i} d={`M${406 + i * 70} 150 H ${418 + i * 70}`} slow marker={false} sw={1.3} />
        ))}

        <Flow d="M722 150 H 742" />
        <Plus x={752} y={150} />

        {/* reconstruction */}
        <Flow d="M763 150 H 800" />
        <Node x={802} y={116} w={118} h={68} title="↑ ×s" sub="pixel-shuffle" emph />
        <Chip x={861} y={206} text="reconstruction" size={10} color={C.mute} anchor="middle" />

        {/* SR */}
        <Flow d="M920 150 H 950" />
        <Img id="hero-sr" x={950} y={85} size={130} href="/figures/monarch-sr.png" />
        <Chip x={1015} y={236} text="SR · ×2 / ×3 / ×4" anchor="middle" color={C.ink} weight={600} />
        <Chip x={1015} y={254} text="sH × sW × 3" anchor="middle" size={10} color={C.mute} />
      </svg>
    </Scroller>
  );
}

/* ─────────────────────────────── motivation ────────────────────────────── */

function Problem() {
  return (
    <Section id="motivation" className="border-b border-border">
      <Reveal>
        <SectionHead
          kicker="the trade-off"
          title={
            <>
              Transformers see far.
              <br />
              CNNs stay light. <span className="text-muted-foreground">SWIFT does both.</span>
            </>
          }
          lead="Self-attention models long-range dependencies that CNNs miss — which is why transformer super-resolution beats convolutional methods on quality. But attention is heavy and slow at inference. SWIFT keeps the long-range modelling and buys back the speed by pushing global context into the frequency domain."
        />
      </Reveal>

      <Reveal className="mt-12">
        <Scroller min={720}>
          <TradeoffDiagram />
        </Scroller>
      </Reveal>

      <div className="mt-12 grid gap-4 md:grid-cols-3">
        {[
          ["Long-range, cheaply", "A Fast Fourier Convolution has an image-wide receptive field in a single layer — global context without the quadratic cost of dense attention."],
          ["Local detail, preserved", "SwinV2 windowed attention with residual post-norm and cosine attention captures fine texture and edges while staying numerically stable to train."],
          ["Small and fast", "At ×4, SWIFT is 596K parameters — 34% smaller than SwinIR — and shaves up to 60% off inference time across the standard benchmarks."],
        ].map(([t, d]) => (
          <Reveal key={t}>
            <div className="h-full rounded-xl border border-border bg-card/30 p-5">
              <div className="text-sm font-semibold">{t}</div>
              <p className="prose mt-2 text-sm leading-relaxed text-muted-foreground">{d}</p>
            </div>
          </Reveal>
        ))}
      </div>
    </Section>
  );
}

function TradeoffDiagram() {
  const C = usePalette();
  return (
    <svg viewBox="0 0 1040 210" className="block h-auto w-full" role="img" aria-label="CNNs are light but local; transformers are global but heavy; SWIFT combines both">
      <Node x={40} y={60} w={220} h={92} title="CNN methods" sub="light · fast · local" />
      <Chip x={150} y={182} text="limited receptive field" anchor="middle" size={10} color={C.mute} />

      <Node x={780} y={60} w={220} h={92} title="Transformers" sub="global · accurate · heavy" />
      <Chip x={890} y={182} text="quadratic attention cost" anchor="middle" size={10} color={C.mute} />

      {/* arrows converging to SWIFT */}
      <Flow d="M260 106 C 372 106, 392 102, 468 102" />
      <Flow d="M780 106 C 668 106, 648 102, 572 102" />

      {/* SWIFT center */}
      <Node x={468} y={56} w={104} h={100} title="SWIFT" sub="best of both" emph />
      <Chip x={520} y={182} text="long-range + lightweight" anchor="middle" size={10} color={C.ink} weight={600} />
    </svg>
  );
}

/* ─────────────────────────────── architecture ──────────────────────────── */

function Architecture() {
  return (
    <Section id="architecture" className="border-b border-border">
      <Reveal>
        <SectionHead
          kicker="architecture"
          title={<>Three stages, one forward pass</>}
          lead="SWIFT follows the canonical SR template — shallow features, deep features, reconstruction — but the deep stage is a stack of four Fourier-Swin Transformer Blocks (FSTB), and a single global residual carries the shallow features all the way to reconstruction."
        />
      </Reveal>

      <Reveal className="mt-12">
        <div className="rounded-2xl border border-border bg-card/20 p-4 md:p-6">
          <Scroller min={1040}>
            <ArchitectureDiagram />
          </Scroller>
        </div>
      </Reveal>

      <div className="mt-10 grid gap-4 md:grid-cols-3">
        {[
          ["01", "Shallow feature extraction", "A single 3×3 convolution lifts the low-resolution RGB image into a 64-channel feature space — a stable, low-level embedding for the transformer stack."],
          ["02", "Deep feature extraction", "Four FSTB blocks followed by a convolution model long-range structure and high-frequency detail. A global residual adds the shallow features back, so the deep stack only has to learn the refinement."],
          ["03", "HQ image reconstruction", "A pixel-shuffle upsampler turns refined features into the ×2 / ×3 / ×4 output. No transposed convolutions, no checkerboard artifacts."],
        ].map(([n, t, d]) => (
          <Reveal key={n}>
            <div className="h-full rounded-xl border border-border bg-card/30 p-5">
              <div className="font-mono text-xs text-muted-foreground">{n}</div>
              <div className="mt-2 text-sm font-semibold">{t}</div>
              <p className="prose mt-2 text-sm leading-relaxed text-muted-foreground">{d}</p>
            </div>
          </Reveal>
        ))}
      </div>
    </Section>
  );
}

function ArchitectureDiagram() {
  const C = usePalette();
  const yMid = 178;
  return (
    <svg viewBox="0 0 1180 360" className="block h-auto w-full" role="img" aria-label="SWIFT three-stage architecture diagram">
      {/* stage band labels */}
      <Chip x={150} y={40} text="01 · SHALLOW" size={10} color={C.mute} anchor="middle" />
      <Chip x={560} y={40} text="02 · DEEP FEATURE EXTRACTION" size={10} color={C.mute} anchor="middle" />
      <Chip x={1000} y={40} text="03 · RECONSTRUCTION" size={10} color={C.mute} anchor="middle" />
      <line x1={300} y1={56} x2={300} y2={330} stroke={C.line} strokeDasharray="2 7" />
      <line x1={820} y1={56} x2={820} y2={330} stroke={C.line} strokeDasharray="2 7" />

      {/* LR */}
      <Img id="arch-lr" x={28} y={138} size={92} href="/figures/monarch-lr.png" />
      <Chip x={74} y={250} text="LR input" anchor="middle" size={10} color={C.sub} />

      {/* shallow conv */}
      <Flow d="M122 184 H 150" />
      <Node x={150} y={154} w={104} h={60} title="Conv 3×3" sub="→ 64-d" />

      {/* deep block container */}
      <rect x={312} y={80} width={496} height={210} rx={12} fill={C.panel} stroke={C.line} strokeDasharray="3 6" />
      <Flow d="M254 184 H 330" />
      {[0, 1, 2, 3].map((i) => (
        <Node key={i} x={330 + i * 86} y={132} w={70} h={104} title="FSTB" sub={`block ${i + 1}`} accent="spatial" />
      ))}
      {[0, 1, 2].map((i) => (
        <Flow key={i} d={`M${400 + i * 86} 184 H ${416 + i * 86}`} slow marker={false} sw={1.3} />
      ))}
      <Flow d="M674 184 H 690" slow marker={false} sw={1.3} />
      <Node x={690} y={132} w={96} h={104} title="Conv" sub="after body" />

      {/* residual skip shallow -> add */}
      <path d="M254 166 C 330 70, 800 70, 858 162" fill="none" stroke={C.mute} strokeWidth={1.5} strokeDasharray="2 5" markerEnd="url(#arr)" />
      <Chip x={556} y={78} text="global residual connection" size={10} color={C.mute} anchor="middle" />

      <Flow d="M808 184 H 858" />
      <Plus x={868} y={yMid - 6 + 8} />

      {/* reconstruction */}
      <Flow d="M880 184 H 916" />
      <Node x={916} y={150} w={132} h={70} title="↑ Upsampler" sub="pixel-shuffle ×s" emph />

      {/* SR */}
      <Flow d="M1048 184 H 1072" />
      <Img id="arch-sr" x={1072} y={132} size={92} href="/figures/monarch-sr.png" />
      <Chip x={1118} y={244} text="SR output" anchor="middle" size={10} color={C.ink} weight={600} />

      {/* legend */}
      <g>
        <rect x={330} y={312} width={3.5} height={12} rx={2} fill={C.ink} />
        <Chip x={342} y={322} text="FSTB = SwinV2 layers + frequency blocks" size={10} color={C.mute} />
      </g>
    </svg>
  );
}

/* ──────────────────────────────── the block ────────────────────────────── */

function Block() {
  return (
    <Section className="border-b border-border">
      <Reveal>
        <SectionHead
          kicker="inside the FSTB"
          title={<>The Fourier-Swin Transformer Block</>}
          lead="Each FSTB interleaves two SwinV2 transformer layers (S2TL) with two Residual Frequency Blocks (RFB), then mixes channels with a 3×3 convolution. A local residual wraps the whole block so gradients flow freely through the deep stack."
        />
      </Reveal>

      <Reveal className="mt-12">
        <div className="rounded-2xl border border-border bg-card/20 p-4 md:p-6">
          <Scroller min={900}>
            <FSTBDiagram />
          </Scroller>
        </div>
      </Reveal>

      <div className="mt-8 flex flex-wrap items-center gap-x-8 gap-y-3 text-xs text-muted-foreground">
        <span className="inline-flex items-center gap-2">
          <span className="inline-block h-3 w-[3px] rounded bg-foreground" /> S2TL — spatial / attention
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="inline-flex h-3 w-[3px] flex-col justify-between">
            <span className="block h-[3px] w-full rounded bg-foreground" />
            <span className="block h-[3px] w-full rounded bg-foreground" />
            <span className="block h-[3px] w-full rounded bg-foreground" />
          </span>
          RFB — frequency / spectral
        </span>
        <span className="font-mono">depths = [2,2,2,2] · rfbs = [2,2,2,2] · dim = 64 · heads = 8 · window = 8</span>
      </div>
    </Section>
  );
}

function FSTBDiagram() {
  const C = usePalette();
  const y = 96;
  const xs = [150, 268, 386, 504];
  return (
    <svg viewBox="0 0 940 240" className="block h-auto w-full" role="img" aria-label="FSTB internals: two SwinV2 transformer layers, two residual frequency blocks, a convolution, and a residual connection">
      <Chip x={40} y={y + 36} text="C" anchor="middle" size={13} color={C.ink} weight={600} />
      <Flow d={`M58 ${y + 30} H ${xs[0]}`} />

      <Node x={xs[0]} y={y} w={100} h={66} title="S2TL" sub="swinV2 #1" accent="spatial" />
      <Flow d={`M${xs[0] + 100} ${y + 33} H ${xs[1]}`} />
      <Node x={xs[1]} y={y} w={100} h={66} title="S2TL" sub="swinV2 #2" accent="spatial" />
      <Flow d={`M${xs[1] + 100} ${y + 33} H ${xs[2]}`} />
      <Node x={xs[2]} y={y} w={100} h={66} title="RFB" sub="freq #1" accent="freq" />
      <Flow d={`M${xs[2] + 100} ${y + 33} H ${xs[3]}`} />
      <Node x={xs[3]} y={y} w={100} h={66} title="RFB" sub="freq #2" accent="freq" />

      <Flow d={`M${xs[3] + 100} ${y + 33} H 648`} />
      <Node x={648} y={y} w={92} h={66} title="Conv" sub="3×3" />
      <Flow d={`M740 ${y + 33} H 800`} />
      <Plus x={812} y={y + 33} />
      <Flow d={`M824 ${y + 33} H 892`} />
      <Chip x={912} y={y + 36} text="C" anchor="middle" size={13} color={C.ink} weight={600} />

      {/* residual skip */}
      <path d={`M90 ${y + 18} C 120 30, 800 30, 812 ${y + 22}`} fill="none" stroke={C.mute} strokeWidth={1.5} strokeDasharray="2 5" markerEnd="url(#arr)" />
      <Chip x={450} y={34} text="local residual" size={10} color={C.mute} anchor="middle" />
    </svg>
  );
}

/* ─────────────────────────────── components ────────────────────────────── */

function Components() {
  return (
    <Section id="components" className="border-b border-border">
      <Reveal>
        <SectionHead kicker="two halves of every block" title={<>Attention for structure. Fourier for reach.</>} />
      </Reveal>

      <div className="mt-12 grid gap-6 lg:grid-cols-2">
        <Reveal>
          <div className="flex h-full flex-col rounded-2xl border border-border bg-card/20 p-5 md:p-6">
            <div className="flex items-center gap-2">
              <span className="inline-block h-3.5 w-[3px] rounded bg-foreground" />
              <h3 className="text-base font-bold">S2TL · SwinV2 Transformer Layer</h3>
            </div>
            <p className="prose mt-2 text-sm leading-relaxed text-muted-foreground">
              Windowed multi-head self-attention with SwinV2&apos;s residual post-norm and cosine attention, plus an
              <span className="text-foreground"> attention-scaling</span> factor that reweights heads. Shifted windows
              alternate every layer so information crosses window boundaries.
            </p>
            <div className="mt-5 rounded-xl border border-border bg-background/60 p-3">
              <Scroller min={440}>
                <S2TLDiagram />
              </Scroller>
            </div>
          </div>
        </Reveal>

        <Reveal>
          <div className="flex h-full flex-col rounded-2xl border border-border bg-card/20 p-5 md:p-6">
            <div className="flex items-center gap-2">
              <span className="inline-flex h-3.5 w-[3px] flex-col justify-between">
                <span className="block h-[3px] w-full rounded bg-foreground" />
                <span className="block h-[3px] w-full rounded bg-foreground" />
                <span className="block h-[3px] w-full rounded bg-foreground" />
              </span>
              <h3 className="text-base font-bold">RFB · Residual Frequency Block</h3>
            </div>
            <p className="prose mt-2 text-sm leading-relaxed text-muted-foreground">
              Channels split in two. A local branch runs dense weight-shared convolutions; a spectral branch sends
              features through a <span className="text-foreground">Fast Fourier Convolution</span> — real FFT, a
              learned filter in frequency space, inverse FFT — for an image-wide receptive field. A spatial-channel
              attention module (SCAM) fuses them.
            </p>
            <div className="mt-5 rounded-xl border border-border bg-background/60 p-3">
              <Scroller min={440}>
                <RFBDiagram />
              </Scroller>
            </div>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

function S2TLDiagram() {
  const C = usePalette();
  const y = 80;
  return (
    <svg viewBox="0 0 624 172" className="block h-auto w-full" role="img" aria-label="SwinV2 transformer layer: windowed attention then layernorm with a residual, followed by an MLP then layernorm with a residual; attention scaling on the attention sublayer">
      <Chip x={20} y={y + 4} text="x" anchor="middle" size={13} color={C.ink} weight={600} />
      <Flow d={`M30 ${y} H 64`} />

      <Node x={64} y={y - 26} w={120} h={52} title="(S)W-MSA" sub="windowed attn" accent="spatial" />
      <Flow d={`M184 ${y} H 206`} />
      <Node x={206} y={y - 26} w={80} h={52} title="LN" sub="post-norm" />
      <Flow d={`M286 ${y} H 300`} />
      <Plus x={312} y={y} />

      <Flow d={`M324 ${y} H 348`} />
      <Node x={348} y={y - 26} w={100} h={52} title="MLP" sub="ratio 1" />
      <Flow d={`M448 ${y} H 470`} />
      <Node x={470} y={y - 26} w={80} h={52} title="LN" sub="post-norm" />
      <Flow d={`M550 ${y} H 564`} />
      <Plus x={576} y={y} />
      <Flow d={`M588 ${y} H 612`} />

      {/* two residual skips — one per SwinV2 sub-block */}
      <path d={`M30 ${y - 12} C 82 26, 268 26, 312 ${y - 11}`} fill="none" stroke={C.mute} strokeWidth={1.3} strokeDasharray="2 5" markerEnd="url(#arr)" />
      <path d={`M324 ${y + 12} C 380 128, 540 128, 576 ${y + 11}`} fill="none" stroke={C.mute} strokeWidth={1.3} strokeDasharray="2 5" markerEnd="url(#arr)" />
      <Chip x={171} y={24} text="residual" size={9} color={C.mute} anchor="middle" />
      <Chip x={450} y={146} text="residual" size={9} color={C.mute} anchor="middle" />
      <Chip x={124} y={y + 44} text="× attention scale" size={9} color={C.mute} anchor="middle" />
    </svg>
  );
}

function RFBDiagram() {
  const C = usePalette();
  const y = 140;
  return (
    <svg viewBox="0 0 500 252" className="block h-auto w-full" role="img" aria-label="Residual frequency block: channels split into a local dense-convolution branch and an FFT-based spectral branch, fused by SCAM attention, then concatenated with a residual connection">
      <Chip x={22} y={y + 4} text="C" anchor="middle" size={13} color={C.ink} weight={600} />
      <Flow d={`M32 ${y} H 56`} marker={false} />
      {/* split node */}
      <circle cx={66} cy={y} r={9} fill={C.panel} stroke={C.edge} />
      <path d={`M62 ${y} H 70 M66 ${y - 4} V ${y + 4}`} stroke={C.ink} strokeWidth={1.2} />

      {/* local branch (top) */}
      <Flow d={`M75 ${y - 5} C 96 78, 104 70, 120 70`} />
      <Node x={120} y={66} w={140} h={48} title="Dense Conv" sub="weight-shared" accent="spatial" />
      <Flow d={`M260 90 C 288 90, 296 124, 314 ${y - 6}`} />

      {/* spectral branch (bottom) — FFC */}
      <Flow d={`M75 ${y + 5} C 96 198, 104 206, 120 206`} />
      <rect x={120} y={172} width={172} height={70} rx={10} fill={C.panel} stroke={C.edge} />
      <Chip x={206} y={188} text="FFC · spectral branch" size={9} color={C.mute} anchor="middle" />
      <text x={136} y={216} fontSize={10} fontFamily="'JetBrains Mono', monospace" fill={C.sub}>FFT</text>
      <g>
        <circle cx={206} cy={212} r={14} fill="none" stroke={C.edge} />
        <circle cx={206} cy={212} r={13} fill="none" stroke={C.mute} className="spin-slow" strokeDasharray="3 5" />
        <circle cx={206} cy={212} r={7.5} fill="none" stroke={C.mute} strokeWidth={0.8} />
        <circle cx={206} cy={212} r={3} fill={C.ink} />
      </g>
      <text x={232} y={216} fontSize={10} fontFamily="'JetBrains Mono', monospace" fill={C.sub}>iFFT</text>
      <text x={174} y={234} fontSize={9} fontFamily="'JetBrains Mono', monospace" fill={C.mute}>BN · ReLU</text>
      <Flow d={`M292 206 C 304 206, 304 160, 314 ${y + 6}`} />

      {/* SCAM fusion */}
      <Node x={314} y={y - 26} w={92} h={52} title="SCAM" sub="attn fuse" emph />
      <Chip x={360} y={y - 38} text="spatial-channel attn" size={9} color={C.mute} anchor="middle" />

      <Flow d={`M406 ${y} H 432`} />
      <Plus x={444} y={y} />
      <Flow d={`M455 ${y} H 482`} />
      <Chip x={496} y={y + 4} text="C" anchor="middle" size={13} color={C.ink} weight={600} />

      {/* residual skip — routed high above both branches */}
      <path d={`M34 ${y - 12} C 60 10, 414 10, 444 ${y - 11}`} fill="none" stroke={C.mute} strokeWidth={1.3} strokeDasharray="2 5" markerEnd="url(#arr)" />
      <Chip x={239} y={42} text="residual" size={9} color={C.mute} anchor="middle" />
    </svg>
  );
}

/* ──────────────────────────────── results ──────────────────────────────── */

type Row = { ds: string; swift: [number, number]; swinir: [number, number] };
const BENCH: Record<"2" | "3" | "4", { params: [number, number]; rows: Row[] }> = {
  "2": {
    params: [579, 878],
    rows: [
      { ds: "Set5", swift: [38.16, 0.9614], swinir: [38.14, 0.9611] },
      { ds: "Set14", swift: [33.86, 0.9207], swinir: [33.86, 0.9206] },
      { ds: "BSD100", swift: [32.29, 0.9012], swinir: [32.31, 0.9012] },
      { ds: "Urban100", swift: [32.6, 0.9328], swinir: [32.76, 0.9328] },
      { ds: "Manga109", swift: [39.15, 0.9784], swinir: [39.12, 0.9783] },
    ],
  },
  "3": {
    params: [600, 886],
    rows: [
      { ds: "Set5", swift: [34.55, 0.9288], swinir: [34.62, 0.9289] },
      { ds: "Set14", swift: [30.57, 0.8464], swinir: [30.54, 0.8463] },
      { ds: "BSD100", swift: [29.21, 0.8082], swinir: [29.2, 0.8082] },
      { ds: "Urban100", swift: [28.61, 0.8612], swinir: [28.66, 0.8624] },
      { ds: "Manga109", swift: [34.18, 0.9483], swinir: [33.98, 0.9478] },
    ],
  },
  "4": {
    params: [596, 897],
    rows: [
      { ds: "Set5", swift: [32.39, 0.8978], swinir: [32.44, 0.8976] },
      { ds: "Set14", swift: [28.82, 0.787], swinir: [28.77, 0.7858] },
      { ds: "BSD100", swift: [27.71, 0.7411], swinir: [27.69, 0.7406] },
      { ds: "Urban100", swift: [26.52, 0.7992], swinir: [26.47, 0.798] },
      { ds: "Manga109", swift: [31.06, 0.9153], swinir: [30.92, 0.9151] },
    ],
  },
};

function Results() {
  const [scale, setScale] = useState<"2" | "3" | "4">("4");
  const data = BENCH[scale];
  return (
    <Section id="results" className="border-b border-border">
      <Reveal>
        <SectionHead
          kicker="quantitative results"
          title={<>SwinIR quality, a third of the weight</>}
          lead="On the five standard benchmarks, SWIFT trades blows with SwinIR on PSNR and SSIM — often winning outright — while carrying far fewer parameters. The frontier chart says it best: SWIFT sits where you want to be, up and to the left."
        />
      </Reveal>

      <div className="mt-12 grid gap-6 lg:grid-cols-5">
        <Reveal className="lg:col-span-3">
          <div className="h-full rounded-2xl border border-border bg-card/20 p-4 md:p-6">
            <div className="mb-2 flex items-baseline justify-between">
              <span className="text-sm font-semibold">PSNR vs parameters</span>
              <span className="font-mono text-[11px] text-muted-foreground">Set5 · ×4</span>
            </div>
            <ScatterChart />
          </div>
        </Reveal>

        <Reveal className="lg:col-span-2">
          <div className="flex h-full flex-col rounded-2xl border border-border bg-card/20 p-4 md:p-6">
            <div className="mb-4 flex items-center justify-between">
              <span className="text-sm font-semibold">Benchmark detail</span>
              <div className="inline-flex rounded-lg border border-border p-0.5">
                {(["2", "3", "4"] as const).map((s) => (
                  <button
                    key={s}
                    onClick={() => setScale(s)}
                    className={`rounded-md px-2.5 py-1 font-mono text-xs transition-colors ${scale === s ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground"}`}
                  >
                    ×{s}
                  </button>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-border bg-border text-center">
              <div className="bg-background px-3 py-3">
                <div className="text-xl font-semibold tabular-nums">{data.params[0]}K</div>
                <div className="text-[11px] text-muted-foreground">SWIFT params</div>
              </div>
              <div className="bg-background px-3 py-3">
                <div className="text-xl font-semibold tabular-nums text-muted-foreground">{data.params[1]}K</div>
                <div className="text-[11px] text-muted-foreground">SwinIR params</div>
              </div>
            </div>

            <table className="mt-3 w-full border-collapse text-xs">
              <thead>
                <tr className="text-muted-foreground">
                  <th className="py-2 text-left font-normal">Dataset</th>
                  <th className="py-2 text-right font-normal">SWIFT</th>
                  <th className="py-2 text-right font-normal">SwinIR</th>
                </tr>
              </thead>
              <tbody className="font-mono tabular-nums">
                {data.rows.map((r) => {
                  const win = r.swift[0] >= r.swinir[0];
                  return (
                    <tr key={r.ds} className="border-t border-border">
                      <td className="py-2 text-left font-sans text-foreground">{r.ds}</td>
                      <td className={`py-2 text-right ${win ? "font-semibold text-foreground" : "text-muted-foreground"}`}>
                        {r.swift[0].toFixed(2)}
                        <span className="ml-1 text-[10px] text-muted-foreground">/{r.swift[1].toFixed(4)}</span>
                      </td>
                      <td className="py-2 text-right text-muted-foreground">
                        {r.swinir[0].toFixed(2)}
                        <span className="ml-1 text-[10px] text-muted-foreground/70">/{r.swinir[1].toFixed(4)}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <p className="mt-3 text-[11px] leading-relaxed text-muted-foreground">
              PSNR (dB) / SSIM, higher is better. Bold = SWIFT ≥ SwinIR. Y-channel, matching the paper&apos;s protocol.
            </p>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

function ScatterChart() {
  const C = usePalette();
  const pts: { name: string; p: number; v: number; emph?: boolean }[] = [
    { name: "CARN", p: 1592, v: 32.13 },
    { name: "IMDN", p: 715, v: 32.21 },
    { name: "ESRT", p: 751, v: 32.19 },
    { name: "SwinIR", p: 897, v: 32.44 },
    { name: "SWIFT", p: 596, v: 32.39, emph: true },
  ];
  const X0 = 70, X1 = 620, Y0 = 320, Y1 = 50;
  const pMin = 500, pMax = 1650, vMin = 32.05, vMax = 32.5;
  const sx = (p: number) => X0 + ((p - pMin) / (pMax - pMin)) * (X1 - X0);
  const sy = (v: number) => Y0 - ((v - vMin) / (vMax - vMin)) * (Y0 - Y1);
  const xticks = [600, 900, 1200, 1500];
  const yticks = [32.1, 32.2, 32.3, 32.4, 32.5];
  return (
    <svg viewBox="0 0 660 380" className="block h-auto w-full" role="img" aria-label="Scatter chart of PSNR versus parameter count; SWIFT achieves near-top PSNR with the fewest parameters">
      {/* gridlines */}
      {yticks.map((t) => (
        <g key={`y${t}`}>
          <line x1={X0} y1={sy(t)} x2={X1} y2={sy(t)} stroke={C.line} />
          <text x={X0 - 8} y={sy(t) + 3} fontSize={10} textAnchor="end" fontFamily="'JetBrains Mono', monospace" fill={C.mute}>
            {t.toFixed(1)}
          </text>
        </g>
      ))}
      {xticks.map((t) => (
        <g key={`x${t}`}>
          <line x1={sx(t)} y1={Y0} x2={sx(t)} y2={Y1} stroke={C.line} />
          <text x={sx(t)} y={Y0 + 18} fontSize={10} textAnchor="middle" fontFamily="'JetBrains Mono', monospace" fill={C.mute}>
            {t}K
          </text>
        </g>
      ))}
      {/* axes */}
      <line x1={X0} y1={Y0} x2={X1} y2={Y0} stroke={C.edge} />
      <line x1={X0} y1={Y0} x2={X0} y2={Y1} stroke={C.edge} />
      <text x={(X0 + X1) / 2} y={368} fontSize={11} textAnchor="middle" fontFamily="'JetBrains Mono', monospace" fill={C.sub}>
        parameters →
      </text>
      <text x={20} y={(Y0 + Y1) / 2} fontSize={11} textAnchor="middle" fontFamily="'JetBrains Mono', monospace" fill={C.sub} transform={`rotate(-90 20 ${(Y0 + Y1) / 2})`}>
        PSNR (dB) ↑
      </text>

      {/* "better" region arrow */}
      <path d={`M${sx(980)} ${sy(32.18)} L ${sx(660)} ${sy(32.36)}`} stroke={C.mute} strokeWidth={1.2} strokeDasharray="2 5" markerEnd="url(#arr)" />
      <Chip x={sx(1000)} y={sy(32.16)} text="fewer params · higher PSNR" size={10} color={C.mute} />

      {/* points */}
      {pts.map((pt) =>
        pt.emph ? (
          <g key={pt.name} filter="url(#soft)">
            <circle cx={sx(pt.p)} cy={sy(pt.v)} r={16} fill="none" stroke={C.hi} strokeWidth={1} opacity={0.5} className="pulse" />
            <circle cx={sx(pt.p)} cy={sy(pt.v)} r={6.5} fill={C.hi} />
            <Chip x={sx(pt.p) + 12} y={sy(pt.v) - 8} text="SWIFT" size={12} color={C.ink} weight={700} />
          </g>
        ) : (
          <g key={pt.name}>
            <circle cx={sx(pt.p)} cy={sy(pt.v)} r={5} fill={C.panel} stroke={C.sub} strokeWidth={1.3} />
            <Chip x={sx(pt.p) + 10} y={sy(pt.v) + 4} text={pt.name} size={11} color={C.sub} />
          </g>
        ),
      )}
    </svg>
  );
}

/* ──────────────────────────────── speed ────────────────────────────────── */

const SPEED: { ds: string; swift: number; swinir: number; pct: number }[] = [
  { ds: "DIV2K Val", swift: 256.48, swinir: 632.8, pct: 60 },
  { ds: "Manga109", swift: 77.07, swinir: 178.56, pct: 57 },
  { ds: "Urban100", swift: 67.77, swinir: 149.11, pct: 55 },
  { ds: "Set14", swift: 28.47, swinir: 48.3, pct: 41 },
  { ds: "BSD100", swift: 24.14, swinir: 38.21, pct: 37 },
  { ds: "Set5", swift: 23.62, swinir: 25.93, pct: 9 },
];

function Speed() {
  const max = 632.8;
  return (
    <Section className="border-b border-border">
      <Reveal>
        <SectionHead
          kicker="inference time · ×4"
          title={<>Faster where it counts</>}
          lead="Wall-clock inference per image at ×4 scale, SWIFT against SwinIR. On the larger images — where super-resolution actually gets used — SWIFT is dramatically quicker, cutting up to 60% of the runtime."
        />
      </Reveal>

      <Reveal className="mt-12">
        <div className="rounded-2xl border border-border bg-card/20 p-5 md:p-7">
          <div className="mb-6 flex items-center gap-5 text-xs text-muted-foreground">
            <span className="inline-flex items-center gap-2">
              <span className="inline-block h-2.5 w-5 rounded-sm bg-foreground" /> SWIFT
            </span>
            <span className="inline-flex items-center gap-2">
              <span className="inline-block h-2.5 w-5 rounded-sm bg-muted" /> SwinIR
            </span>
            <span className="ml-auto font-mono">ms · lower is better</span>
          </div>

          <div className="flex flex-col gap-5">
            {SPEED.map((s) => (
              <div key={s.ds} className="flex items-center gap-3 sm:gap-4">
                <div className="w-16 shrink-0 text-right font-mono text-[11px] text-muted-foreground sm:w-24 sm:text-xs">{s.ds}</div>
                <div className="min-w-0 flex-1">
                  <div className="h-5 rounded-sm bg-muted" style={{ width: `${Math.max(2, (s.swinir / max) * 100)}%` }} />
                  <div className="mt-1.5 h-5 rounded-sm bg-foreground" style={{ width: `${Math.max(2, (s.swift / max) * 100)}%` }} />
                </div>
                <div className="hidden w-28 shrink-0 text-right font-mono text-xs sm:block">
                  <span className="font-semibold tabular-nums text-foreground">{s.swift.toFixed(0)}</span>
                  <span className="text-muted-foreground"> / {s.swinir.toFixed(0)} ms</span>
                </div>
                <div className="w-12 shrink-0 text-right font-mono text-sm font-semibold tabular-nums sm:w-14">▼{s.pct}%</div>
              </div>
            ))}
          </div>
        </div>
      </Reveal>
    </Section>
  );
}

/* ─────────────────────────────── qualitative ───────────────────────────── */

function Qualitative() {
  return (
    <Section className="border-b border-border">
      <Reveal>
        <SectionHead
          kicker="qualitative comparison"
          title={<>Sharper edges, fewer artifacts</>}
          lead="On the highlighted patches, SWIFT recovers lattice structure and text strokes that lighter CNN methods blur away — recovering detail competitive with the much heavier transformer baselines."
        />
      </Reveal>

      <Reveal className="mt-10">
        <figure className="overflow-hidden rounded-2xl border border-border bg-card/20 p-2 md:p-3">
          <img src="/figures/qualitative.png" alt="Qualitative comparison of SWIFT against state-of-the-art super-resolution methods on cropped image patches" className="w-full rounded-xl" loading="lazy" />
          <figcaption className="px-2 py-3 text-center font-mono text-[11px] text-muted-foreground">
            Qualitative results on ×4 super-resolution. Red box marks the compared region. Figure from the paper.
          </figcaption>
        </figure>
      </Reveal>
    </Section>
  );
}

/* ──────────────────────────────── recipe ───────────────────────────────── */

function Recipe() {
  const spec: [string, string][] = [
    ["Training data", "DIV2K — 800 HR images"],
    ["Iterations", "700K per scale, from scratch"],
    ["Optimizer", "Adam · lr 2e-4 · cosine decay"],
    ["Batch / patch", "64 · 128/192/256 px (×2/×3/×4)"],
    ["Embedding dim", "64 · 8 heads · window 8"],
    ["Hardware", "single NVIDIA A100 · mixed precision"],
  ];
  return (
    <Section className="border-b border-border">
      <div className="grid gap-10 lg:grid-cols-2">
        <Reveal>
          <SectionHead
            kicker="training recipe"
            title={<>Trained once, per scale, from scratch</>}
            lead="Unlike methods that pre-train ×2 and fine-tune the rest, every SWIFT scale is trained end-to-end from scratch on DIV2K. The spec is deliberately modest — one A100, mixed precision, no tricks."
          />
          <div className="mt-8 overflow-hidden rounded-xl border border-border">
            {spec.map(([k, v], i) => (
              <div key={k} className={`flex items-baseline justify-between gap-4 px-4 py-3 ${i % 2 ? "bg-card/20" : "bg-background"}`}>
                <span className="text-xs text-muted-foreground">{k}</span>
                <span className="text-right font-mono text-xs text-foreground">{v}</span>
              </div>
            ))}
          </div>
        </Reveal>

        <Reveal>
          <Terminal
            title="train.py"
            lines={[
              { p: "$", t: "python3 train.py --scale=4 --patch_size=256 \\" },
              { t: "    --root=./Datasets --lr=2e-4 --batch_size=64 \\" },
              { t: "    --n_epochs=100000 --threads=8 --model=SWIFTx4 \\" },
              { t: "    --cuda --amp --load_mem", c: true },
              { t: "" },
              { o: "▸ loading DIV2K (800 imgs) → RAM" },
              { o: "▸ SWIFT ×4 · 596K params · AMP on" },
              { o: "▸ iter 700000/700000  PSNR 32.39  SSIM 0.8978" },
              { o: "✓ checkpoint → ./model_zoo/SWIFT/SWIFT-S-4x.pth" },
            ]}
          />
        </Reveal>
      </div>
    </Section>
  );
}

/* ─────────────────────────────── quickstart ────────────────────────────── */

function Quickstart() {
  return (
    <Section className="border-b border-border">
      <div className="grid gap-10 lg:grid-cols-2">
        <Reveal>
          <SectionHead
            kicker="run it"
            title={<>From checkpoint to upscaled image</>}
            lead="Pretrained ×2 / ×3 / ×4 checkpoints ship in the repo. Run a prediction directly, or serve the model with TorchServe behind an HTTP endpoint — both come pre-baked as Docker images."
          />
          <ul className="prose mt-8 space-y-3 text-sm text-muted-foreground">
            {[
              ["predict.py", "batch inference on a folder, with optional PSNR/SSIM against ground truth"],
              ["TorchServe", "production HTTP inference, CPU or GPU, via prebuilt images"],
              ["JIT + tiling", "--jit to compile, --forward_chop to fit large images in limited memory"],
            ].map(([a, b]) => (
              <li key={a} className="flex gap-3">
                <span className="mt-1 inline-block h-1.5 w-1.5 shrink-0 rounded-full bg-foreground" />
                <span>
                  <span className="font-mono text-foreground">{a}</span> — {b}
                </span>
              </li>
            ))}
          </ul>
        </Reveal>

        <Reveal>
          <Terminal
            title="inference"
            lines={[
              { o: "# 1 · predict on a folder of low-res images" },
              { p: "$", t: "python3 predict.py --scale=4 \\" },
              { t: "    --model_path=model_zoo/SWIFT/SWIFT-S-4x.pth \\" },
              { t: "    --folder_lq=./inputs --cuda --jit", c: true },
              { o: "✓ results/ ← super-resolved ×4" },
              { t: "" },
              { o: "# 2 · or serve it over HTTP (TorchServe)" },
              { p: "$", t: "docker run --gpus all -p 8080:8080 \\" },
              { t: "    ivishalr/swift:latest-gpu", c: true },
              { p: "$", t: "python3 serve/infer.py --path=img.png --scale=4" },
            ]}
          />
        </Reveal>
      </div>
    </Section>
  );
}

/* The terminal is deliberately dark in both themes (a code-block convention),
   so its text colours are fixed rather than theme-token based. */
function Terminal({ title, lines }: { title: string; lines: { p?: string; t?: string; o?: string; c?: boolean }[] }) {
  return (
    <div className="overflow-hidden rounded-xl border border-zinc-800 bg-[#0a0a0a] shadow-sm">
      <div className="flex items-center gap-2 border-b border-zinc-800 px-4 py-2.5">
        <span className="h-3 w-3 rounded-full bg-[#3a3a40]" />
        <span className="h-3 w-3 rounded-full bg-[#2a2a2e]" />
        <span className="h-3 w-3 rounded-full bg-[#222226]" />
        <span className="ml-2 font-mono text-[11px] text-zinc-500">{title}</span>
      </div>
      <pre className="overflow-x-auto px-4 py-4 font-mono text-[12px] leading-relaxed">
        {lines.map((l, i) => (
          <div key={i} className="whitespace-pre">
            {l.o !== undefined ? (
              <span className="text-zinc-500">{l.o}</span>
            ) : (
              <>
                {l.p && <span className="select-none text-zinc-600">{l.p} </span>}
                <span className={l.c ? "text-zinc-500" : "text-zinc-100"}>{l.t}</span>
              </>
            )}
          </div>
        ))}
      </pre>
    </div>
  );
}

/* ──────────────────────────────── citation ─────────────────────────────── */

const BIBTEX = `@article{ramesha2024swift,
  author  = {Vishal Ramesha and Yashas Kadambi and B. S. Abhishek Aditya and
             T. Vijay Prashant and S. S. Shylaja},
  title   = {Toward Faster and Efficient Lightweight Image Super-Resolution
             Using Transformers and Fourier Convolutions},
  journal = {Artificial Intelligence and Applications},
  year    = {2024},
  volume  = {3},
  number  = {2},
  pages   = {168--178},
  doi     = {10.47852/bonviewAIA42021930},
}`;

function CopyButton({ text }: { text: string }) {
  const [done, setDone] = useState(false);
  return (
    <button
      onClick={() => {
        navigator.clipboard?.writeText(text).then(() => {
          setDone(true);
          setTimeout(() => setDone(false), 1600);
        });
      }}
      className="inline-flex items-center gap-1.5 rounded-md border border-border px-2.5 py-1 font-mono text-[11px] text-muted-foreground transition-colors hover:text-foreground"
    >
      {done ? "copied ✓" : "copy"}
    </button>
  );
}

function Cite() {
  return (
    <Section id="paper" className="border-b border-border">
      <div className="grid gap-10 lg:grid-cols-2">
        <Reveal>
          <SectionHead
            kicker="the paper"
            title={<>Toward Faster and Efficient Lightweight Image Super-Resolution</>}
            lead="Published in Artificial Intelligence and Applications (2024). A hybrid of transformers and Fast Fourier Convolutions for lightweight single-image super-resolution. If SWIFT is useful in your work, a citation is appreciated."
          />
          <div className="mt-8 flex flex-col gap-3">
            <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Authors · PES University</div>
            <div className="flex flex-wrap gap-2">
              {AUTHORS.map(([name, url]) => (
                <a key={name} href={url} className="rounded-lg border border-border px-3 py-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground">
                  {name}
                </a>
              ))}
            </div>
            <div className="mt-3 flex flex-wrap gap-3">
              <a href={PAPER_PDF} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 rounded-lg bg-foreground px-4 py-2.5 text-sm font-medium text-background transition-opacity hover:opacity-90">
                <PaperIcon /> Read the paper
              </a>
              <a href={PAPER_URL} target="_blank" rel="noreferrer" className="inline-flex items-center gap-2 rounded-lg border border-border px-4 py-2.5 text-sm text-muted-foreground transition-colors hover:text-foreground">
                Publisher · DOI
              </a>
              <a href={REPO} className="inline-flex items-center gap-2 rounded-lg border border-border px-4 py-2.5 text-sm text-muted-foreground transition-colors hover:text-foreground">
                <GitHubMark /> Code
              </a>
            </div>
          </div>
        </Reveal>

        <Reveal>
          <div className="overflow-hidden rounded-xl border border-border bg-card/20">
            <div className="flex items-center justify-between border-b border-border px-4 py-2.5">
              <span className="font-mono text-[11px] text-muted-foreground">citation.bib</span>
              <CopyButton text={BIBTEX} />
            </div>
            <pre className="overflow-x-auto px-4 py-4 font-mono text-[11.5px] leading-relaxed text-muted-foreground">{BIBTEX}</pre>
          </div>
        </Reveal>
      </div>
    </Section>
  );
}

/* ──────────────────────────────── footer ───────────────────────────────── */

function Footer() {
  return (
    <footer className="relative overflow-hidden">
      <div className="pointer-events-none absolute inset-x-0 -bottom-10 select-none text-center text-[24vw] font-bold leading-none tracking-tighter text-foreground/[0.035]">
        SWIFT
      </div>
      <div className="relative mx-auto w-full max-w-6xl px-5 py-14">
        <div className="flex flex-col gap-8 md:flex-row md:items-start md:justify-between">
          <div className="max-w-sm">
            <div className="flex items-center gap-2.5">
              <Logo size={24} />
              <span className="text-sm font-semibold tracking-[0.22em]">SWIFT</span>
            </div>
            <p className="prose mt-3 text-sm leading-relaxed text-muted-foreground">
              Lightweight image super-resolution with SwinV2 transformers and Fast Fourier Convolutions.
            </p>
          </div>
          <div className="grid grid-cols-2 gap-10 text-sm sm:grid-cols-3">
            <div className="flex flex-col gap-2">
              <span className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Project</span>
              <a href={REPO} className="text-muted-foreground transition-colors hover:text-foreground">GitHub</a>
              <a href="#architecture" className="text-muted-foreground transition-colors hover:text-foreground">Architecture</a>
              <a href="#results" className="text-muted-foreground transition-colors hover:text-foreground">Results</a>
            </div>
            <div className="flex flex-col gap-2">
              <span className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Paper</span>
              <a href={PAPER_PDF} target="_blank" rel="noreferrer" className="text-muted-foreground transition-colors hover:text-foreground">Read PDF</a>
              <a href={PAPER_URL} target="_blank" rel="noreferrer" className="text-muted-foreground transition-colors hover:text-foreground">Publisher</a>
              <a href="#paper" className="text-muted-foreground transition-colors hover:text-foreground">Citation</a>
            </div>
            <div className="flex flex-col gap-2">
              <span className="text-xs uppercase tracking-[0.18em] text-muted-foreground">License</span>
              <span className="text-muted-foreground">MIT</span>
            </div>
          </div>
        </div>
        <div className="mt-10 border-t border-border pt-6 font-mono text-[11px] text-muted-foreground">
          SWIFT · Vishal, Abhishek, Yashas, Vijay, Shylaja · PES University · 2024
        </div>
      </div>
    </footer>
  );
}
