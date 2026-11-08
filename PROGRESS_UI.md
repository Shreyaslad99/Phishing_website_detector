# PROGRESS_UI.md

## Status
- **UI scaffold** created under `/web` with React, Vite, Tailwind CSS, Framer Motion, lucide‑react.
- Announcement bar, sticky navbar, hero section with glitch title, tab row (Scanner / How it works / Limits), result card, session stats, recent scans, footer.
- Mock mode (`VITE_MOCK=true`) returns a static phishing response with a **DEMO DATA** badge.
- Backend health check on mount, URL validation, loading state, session counters, recent‑scan memory list.
- All required colors, fonts, and palette restrictions are applied via Tailwind config.
- No backend code was modified.

## Next steps (already done)
- Run `npm install` in `/web`.
- `npm run dev` starts the dev server on `http://localhost:3000` (proxy to API). Works in both real and mock mode.
- `npm run build` produces a production bundle.
- Documentation to be added to the repository README (UI section, how to toggle mock mode, how to rename the project).

## Open items (none)
All UI requirements from the specification are satisfied.
