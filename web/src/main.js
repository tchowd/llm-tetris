import './style.css'
import referenceBlocks from './blocks/catalog.json'
import { blockSvg, compositionSvg, normalizeBlocks } from './blocks/render.js'
import { setupAssembly } from './assemble.js'

const blocks = normalizeBlocks(referenceBlocks)

const app = document.querySelector('#app')
const params = new URLSearchParams(location.search)
const block = blocks.find(({ id }) => id === params.get('block'))
const gallery = params.get('view') === 'blocks'

if (block) {
  document.title = `Tetra — Block ${block.id}`
  app.className = 'single-view'
  app.innerHTML = `<main class="single-block" style="--block-width: ${block.width}px">${blockSvg(block)}</main>`
} else if (gallery) {
  app.className = 'library-view'
  app.innerHTML = `
    <header class="library-header">
      <a href="./" class="wordmark">TETRA</a>
      <h1>The building blocks.</h1>
      <p>${blocks.length} vector studies. Three colors. One composition.</p>
      <a href="./">View composition <span aria-hidden="true">↗</span></a>
    </header>
    <main class="block-library" aria-label="All reference blocks">
      ${blocks.map(item => `<a class="block-card" href="?block=${item.id}" aria-label="Open block ${item.id}">
        <div class="block-art">${blockSvg(item)}</div>
        <span class="block-label">BLOCK ${item.id.padStart(2, '0')}<span>${item.width} × ${item.height} <span aria-hidden="true">↗</span></span></span>
      </a>`).join('')}
    </main>`
} else {
  app.className = 'composition-view'
  app.innerHTML = `<main class="composition" aria-label="Tetra geometric composition">${compositionSvg(blocks)}</main>`
  setupAssembly(app.querySelector('.composition'), blocks)
}

app.insertAdjacentHTML('beforeend', `
  <details class="view-menu">
    <summary aria-label="Choose artwork view" title="Choose artwork view">
      <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true"><path d="M1 1h6v6H1zM9 1h6v6H9zM1 9h6v6H1zM9 9h6v6H9z" fill="currentColor"/></svg>
    </summary>
    <nav aria-label="Artwork views">
      <a href="./" ${!block && !gallery ? 'aria-current="page"' : ''}>Composition</a>
      <a href="?block=1" ${block?.id === '1' ? 'aria-current="page"' : ''}>First block</a>
      <a href="?view=blocks" ${gallery ? 'aria-current="page"' : ''}>All ${blocks.length} blocks</a>
    </nav>
  </details>`)

document.addEventListener('keydown', event => {
  if (event.key === 'Escape') document.querySelector('.view-menu').open = false
})
