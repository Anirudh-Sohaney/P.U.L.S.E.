# PULSE landing and shared navigation

The hero places a compact pharmacy stock animation beside “Local care. A clearer plan.” The demo introduction and actions sit below this row. On narrow screens these elements stack without horizontal scrolling. The animation includes manual stage controls, pauses on hover or focus, and respects reduced motion.

Landing-page sections and descriptive copy follow the original website at commit `a58cfb0b453793997f30c23fb02e35d3cacbca33`. The requested headline, pharmacy animation, and demo entry point remain. The visual system uses a porcelain background, graphite text, muted blue accents, self-hosted Instrument Serif headings, and Manrope interface text.

The public pages and workspace share one header. `/login` and `/signup` use the repository's existing cookie-session API and browser-plan unlock flow. Logout and logout-everywhere preserve local locking and server session termination. The original `/dashboard`, browser forecasting, encrypted storage, signals, API guide, and backend remain in place. `/demo` provides an isolated interactive sample; its data is illustrative.

## Validation

From `website/frontend`:

```sh
npm ci
npx playwright install chromium
npm run lint
npm run build
npm run test:e2e
```

The browser checks cover landing/demo/access widths from 320 to 1440 pixels, hero placement, manual animation stages, source-copy sections, demo filtering, and the cookie-session login/local unlock/logout flow with mocked API responses. They do not validate a live backend or forecast training.
