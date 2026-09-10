import { readFile, writeFile } from 'node:fs/promises'
import { blockSvg, compositionSvg, normalizeBlocks } from '../src/blocks/render.js'

const blocks = JSON.parse(await readFile(new URL('../src/blocks/catalog.json', import.meta.url), 'utf8'))
await writeFile(new URL('../../assets/blocks/tetris1.svg', import.meta.url), compositionSvg(blocks) + '\n')
for (const block of normalizeBlocks(blocks)) {
  await writeFile(new URL(`../../assets/blocks/block-${block.id}.svg`, import.meta.url), blockSvg(block) + '\n')
}
console.log('Exported the composition and all blocks at a uniform 230 × 327 size')
