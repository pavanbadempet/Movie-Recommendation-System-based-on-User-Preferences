# Accessibility (WCAG 2.1 AA)

This document records the accessibility implementation decisions and known status of the APEX React frontend.

---

## Implemented

### Semantic HTML & ARIA
- `<main>`, `<header>`, `<section>`, `<article>`, `<aside>` landmark elements used throughout
- `role="dialog"` and `aria-modal="true"` on `MovieDialog`
- `aria-label` on dialog (`aria-label={movie.title + " details"}`)
- `aria-label` on feedback button group (`role="group" aria-label="Feedback for {movie.title}"`)
- `aria-label` on home navigation grid (`aria-label="Application navigation"`)
- `aria-label` on result context bar (`aria-label="Result context"`)
- `aria-label` on rating circle (`aria-label="Rating {value} out of 10"`)
- `aria-label` on video toggle button (`aria-label="Pause/Play trailer"`)
- `<span className="visually-hidden">` for screen-reader-only text on icon-only buttons

### Keyboard Navigation
- All interactive elements are `<button>` or `<a>` — no `div` click handlers
- `Escape` key closes `MovieDialog` (via `keydown` listener)
- Focus trap: `MovieDialog` uses `aria-modal="true"` which signals modal boundary to assistive tech
- `document.body.classList.add("modal-open")` prevents background scroll when dialog is open

### Focus Management
- `openSearch()` calls `input.focus()` after navigation to search page
- Title select dropdown uses `onBlur` with `contains()` check to close without stealing focus

### Color & Contrast
- Dark theme with light text — primary text `#f8fafc` on `#0a0a0f` background exceeds 4.5:1 ratio
- Rating circle uses dynamic color (`#21d07a` green / `#d2d531` yellow / `#db2360` red) — supplemented by numeric value so color is not the sole indicator
- Status badges use both icon and text label — not color-only

### Images
- All `<img>` elements have `alt` attributes
- Poster images use `alt={movie.title}`
- Decorative backdrop images use `alt=""`
- `loading="lazy"` on all non-hero images

### Motion
- Spinning loader uses CSS `animation` — respects `prefers-reduced-motion` via `styles.css`
- Trailer autoplay is muted and has a pause/play toggle

---

## Known Gaps & Remediation Plan

| Issue | Severity | Plan |
|-------|----------|------|
| Focus is not explicitly moved into `MovieDialog` on open | Medium | Add `useEffect` that calls `dialogRef.current?.focus()` on mount |
| Title select dropdown has no `role="listbox"` / `role="option"` | Medium | Refactor to use `<ul role="listbox">` with `<li role="option">` |
| Trailer `<iframe>` title is set but `youtube-nocookie.com` content is not fully accessible | Low | Acceptable — iframe is decorative; poster fallback shown when no trailer |
| Admin dashboard (`frontend/admin_dashboard.html`) has no ARIA landmarks | Low | Add `<main>`, `<nav>` landmarks in next admin UI iteration |
| No skip-to-main-content link | Low | Add `<a href="#main-content" class="skip-link">Skip to content</a>` |

---

## Testing

Full WCAG 2.1 AA validation requires manual testing with assistive technologies. Automated tools catch ~30–40% of issues.

Recommended tools:
- **axe DevTools** browser extension — run on each page
- **NVDA + Firefox** (Windows) or **VoiceOver + Safari** (macOS) — screen reader smoke test
- **Lighthouse accessibility audit** — `bun run build && bunx serve -s dist -l 5173` then run Lighthouse

To run axe in CI (once `@axe-core/playwright` is added):
```bash
npx playwright test --grep accessibility
```
