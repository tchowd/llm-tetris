# Tetra vector artwork

Run `npm install` then `npm run dev` from this directory.

- `/` assembles the `assets/tetris1.png` motifs on a uniform seven-column, three-row grid. Every block is 230 × 327, and all row and column seams align.
- `/?block=1` shows the first requested block at its original 230 × 327 size.
- `/?view=blocks` shows all 21 supplied block references, including duplicate motifs and narrow crops. Select any block to open it individually.
- The small bottom-left menu switches views. Escape closes it.
- On page entry, the animation automatically slides the central motif pieces horizontally and vertically into a centered pixel-letter TETRA wordmark. Every tile retains its original background color. Letter strokes use contrasting green, black, or white, changing color at tile seams. There is no Start or Replay button; reloading the page runs the entrance again. The final word stays centered when resized; reduced-motion preferences skip the movement.
- During the entrance, pairs of surrounding tiles ripple outward from the center. Afterward, one tile (occasionally two) slides its foreground by one cell and returns, with a quiet pause between moves. At most two surrounding tiles animate at once. Their backgrounds and the central word stay fixed. Surrounding motion stops in hidden tabs and when reduced motion is enabled.
- As TETRA forms, the mosaic and the backgrounds behind the letters fade to muted, low-contrast gray over the same two seconds. Only the letter shapes retain their full green, black, and white colors. Reloading restores the original palette before the entrance. Reduced motion applies the color change immediately.
- Play, Read, and Data appear at the bottom center after both the letter animation and gray fade finish. All three slide down to their section with the same 900ms transition. The Tetra wordmark in each section slides back up to the landing page. Browser Back/Forward and direct `#play`, `#read`, and `#data` links are supported.
- Play reserves the board and score layout for the future game; gameplay is not implemented. Read contains clearly labeled draft previews, category filters, and article layouts. Data contains empty metrics and chart areas, model/suite filters, a model register, and a downloadable CSV header template. No evaluation results or backend connections are fabricated.

The full artwork fills the viewport and crops proportionally on narrow screens. Every individual block and SVG export uses the same 230 × 327 dimensions. Narrow reference crops use their matching full motif (15-2 and 22 use 3; 23 uses 9), so they fill a complete tile without stretching a half-width shape. There are no reference PNGs, external fonts, or network dependencies in the rendered artwork.

## Edit the artwork

`src/blocks/catalog.json` records each reference filename, original dimensions, background color, and rectangles as `[x, y, width, height, color]`. Edit these coordinates to change a shape. `src/blocks/render.js` contains the reusable SVG renderer and the full composition's placement coordinates. CSS custom properties `--tetra-green`, `--tetra-black`, and `--tetra-white` control the palette.

The original PNGs remain untouched. Standalone vector files for every reference are in `../assets/blocks/block-*.svg`. `../assets/blocks/tetris1.svg` is the assembled vector artwork. The reconstruction preserves the geometry in flat colors; it omits the reference's raster grain and compression artifacts.

## Rebuild vector exports

`python3 scripts/rebuild-blocks.py` regenerates the catalog and individual SVGs from the original PNGs (requires Pillow and NumPy). It classifies the three color regions, detects long orthogonal edges, and merges adjacent rectangles. It checks that each reconstruction agrees with at least 99% of its source's classified color regions. Hand edits to generated files will be replaced by this command.

`node scripts/export-composition.js` exports all normalized block SVGs and the standalone mosaic SVG. Run this after changing the catalog or renderer. The rebuild script also runs this export automatically after checking the original reference geometry.

`npm run build` creates the production site in `dist/`.

`src/assemble.js` defines the pixel alphabet, extracts the visible source pieces from the composition geometry, and animates them into the wordmark. Adjust the alphabet or motion timing there.

`src/surroundings.js` controls the outward ripple and occasional puzzle motion. Each tile clips its moving foreground over its fixed background; a central cutout protects TETRA. The scheduler waits for each batch to finish before starting another, and resets on resize.

`src/pages.js` and `src/pages.css` implement the three sections, shared navigation, sample article content, empty analytics template, and page transition. The landing animation pauses its surrounding motion while a section is open. Hidden sections are inert, keyboard focus follows navigation, and reduced motion skips the slide.
