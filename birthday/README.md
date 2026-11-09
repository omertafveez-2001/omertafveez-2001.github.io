# A little birthday surprise

A standalone, five-chapter birthday gift. Nothing links it from the main portfolio. All source and exported files live in this folder.

## Open the dummy page

The built page is `birthday/index.html`. From the repository root, run:

```sh
python3 -m http.server 8000
```

Then open **http://localhost:8000/birthday/**. Serving over HTTP is recommended for the Next.js scripts. Once this folder is published to the existing GitHub Pages site, the address will be **https://omertafveez-2001.github.io/birthday/**. Creating these files does not publish them; commit and push when ready. This is an unlisted URL, not a private or password-protected page. The page asks search engines not to index it.

## Customize

Edit `birthdayConfig.ts`:

- `nickname`, `greeting`, `welcome`: the first screen.
- `favoriteSound`: guesses and the sweet reveal.
- `routine`: captions, all five button responses, and the ending.
- `quiz.words`: friendly sample Roman Urdu words. Replace these with the words you actually taught her. Each item has a `word`, `pronunciation`, three `options`, a zero-based `correct` index, and an `explanation`. Keep at least one question.
- `letter`: the invitation, poem (one string per stanza, `\n` for line breaks), and signature.
- `colors`, `decorations`: theme colors and optional confetti.

The supplied personal memories and poem are from the brief. Quiz vocabulary is sample content. The nickname is a placeholder. No recordings, external images, API, database, or autoplay audio are used; illustrations and fonts work offline.

After editing, rebuild to update `index.html` and `assets/`. The `.ts` source does not change the exported page until rebuilt.

## Development

Requires Node.js 20.9+.

```sh
cd birthday
npm ci
npm run dev
```

Open http://localhost:3000. Built with Next.js, TypeScript, Tailwind CSS, Framer Motion, and Lucide React. The main experience is in `app/page.tsx`, theme/layout in `app/globals.css`, and original SVG artwork in `components/Illustrations.tsx`.

## Build

```sh
npm run typecheck
npm run build
```

Next.js exports into ignored `out/`. `scripts/export.mjs` copies the entry pages and script/style bundles alongside the source, changing the `_next/` folder name to `assets/` so GitHub Pages/Jekyll can serve it without changing the main site configuration. Commit `index.html`, `404.html`, and `assets/` along with the source. No existing root files need changes. Rebuilding replaces `assets/` so old bundles are not retained.

## Vercel

Import the repository, set **Root Directory** to `birthday`, select the Next.js preset, use `npm run build`, and leave the Output Directory as the preset default. Alternatively select the Other preset with build command `npm run build` and Output Directory `out` to deploy the untouched Next.js static export. No environment variables are required.

## Accessibility and interaction

Only the current chapter is shown. Back navigation and unlocked chapter indicators preserve progress. The sound question accepts any guess, the routine unlocks after five taps, quiz feedback gives the right answer on every attempt, and the final envelope opens and closes. Replay resets the experience. Buttons support keyboard navigation, dynamic answers are announced, and motion follows the device’s reduced-motion preference. No personal progress is stored or sent anywhere.
