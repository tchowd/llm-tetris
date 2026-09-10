import { palette } from './blocks/render.js'

// Only foreground groups move. Tile fields, clips and the central word stay fixed.
export function setupSurroundings(container, blocks, getLayout, reducedMotion) {
  const byId = new Map(blocks.map(block => [block.id, block]))
  const active = new Set()
  let layer, candidates = [], timer, generation = 0, started = false, turn = 0

  function cancel() {
    generation++
    clearTimeout(timer)
    for (const animation of active) animation.cancel()
    active.clear()
    layer?.querySelectorAll('[data-moving]').forEach(node => node.removeAttribute('data-moving'))
  }

  function build() {
    layer?.remove()
    const { bounds, tiles } = getLayout()
    layer = document.createElementNS('http://www.w3.org/2000/svg', 'svg')
    layer.classList.add('surroundings')
    layer.setAttribute('viewBox', `0 0 ${innerWidth} ${innerHeight}`)
    layer.setAttribute('aria-hidden', 'true')
    layer.setAttribute('shape-rendering', 'crispEdges')
    // The center cutout also protects the word on narrow screens and at tile seams.
    const outside = `M0 0H${innerWidth}V${innerHeight}H0Z M${bounds.x} ${bounds.y}v${bounds.height}h${bounds.width}v${-bounds.height}Z`
    layer.innerHTML = `<defs><clipPath id="surrounding-window"><path d="${outside}" clip-rule="evenodd"/></clipPath></defs><g clip-path="url(#surrounding-window)"></g>`
    const group = layer.lastElementChild
    candidates = []
    for (const tile of tiles) {
      const cx = tile.x + tile.width / 2, cy = tile.y + tile.height / 2
      const inside = cx > bounds.x && cx < bounds.x + bounds.width && cy > bounds.y && cy < bounds.y + bounds.height
      if (inside || tile.x >= innerWidth || tile.y >= innerHeight || tile.x + tile.width <= 0 || tile.y + tile.height <= 0) continue
      const block = byId.get(tile.id)
      group.insertAdjacentHTML('beforeend', `<svg data-surrounding-tile="${tile.id}" x="${tile.x}" y="${tile.y}" width="${tile.width}" height="${tile.height}" viewBox="0 0 ${block.width} ${block.height}" preserveAspectRatio="none" overflow="hidden">
        <rect class="surrounding-background" width="${block.width}" height="${block.height}" fill="${palette[block.background]}"/>
        <g class="surrounding-shapes">${block.rects.map(([x, y, width, height, color]) => `<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${palette[color]}"/>`).join('')}</g>
      </svg>`)
      const element = group.lastElementChild.querySelector('.surrounding-shapes')
      const dx = cx - innerWidth / 2, dy = cy - innerHeight / 2
      candidates.push({ element, id: tile.id, distance: Math.hypot(dx, dy),
        horizontal: Math.abs(dx) > Math.abs(dy), sign: Math.sign(Math.abs(dx) > Math.abs(dy) ? dx : dy) || 1,
        step: block.width / 6 })
    }
    candidates.sort((a, b) => a.distance - b.distance)
    // Keep the surroundings below the separate central letter animation.
    container.insertBefore(layer, container.querySelector('.assembly'))
  }

  async function move(tile, { ripple = false, index = 0 } = {}) {
    const horizontal = ripple ? tile.horizontal : (turn + index) % 2 === 0
    const amount = tile.step * (ripple ? .45 * tile.sign : (turn % 2 ? -1 : 1))
    const shift = horizontal ? `translate(${amount}px, 0px)` : `translate(0px, ${amount}px)`
    tile.element.dataset.moving = ripple ? 'ripple' : 'puzzle'
    const animation = tile.element.animate([
      { transform: 'translate(0px, 0px)', offset: 0 },
      { transform: shift, offset: .45 },
      { transform: shift, offset: .6 },
      { transform: 'translate(0px, 0px)', offset: 1 },
    ], { duration: ripple ? 240 : 1600, easing: 'cubic-bezier(.45, 0, .25, 1)' })
    active.add(animation)
    try { await animation.finished } catch { /* Reset, resize or a hidden tab cancels the move. */ }
    finally {
      active.delete(animation)
      tile.element.removeAttribute('data-moving')
    }
  }

  const allowed = () => started && !document.hidden && !reducedMotion.matches

  function scheduleIdle(token) {
    if (token !== generation || !allowed() || !candidates.length) return
    timer = setTimeout(async () => {
      if (token !== generation || !allowed()) return
      // One tile, then occasionally a pair; the next batch waits for completion.
      const count = turn % 3 === 2 && candidates.length > 1 ? 2 : 1
      const first = (turn * 5) % candidates.length
      const batch = Array.from({ length: count }, (_, i) => candidates[(first + i * Math.max(1, Math.floor(candidates.length / 2))) % candidates.length])
      await Promise.all(batch.map((tile, index) => move(tile, { index })))
      turn++
      scheduleIdle(token)
    }, 2600 + turn % 3 * 650)
  }

  async function ripple(token) {
    // Nearest pair first, then successive pairs farther from the word. At most two
    // surrounding tiles are moving at any instant, including during the ripple.
    for (let i = 0; i < candidates.length; i += 2) {
      if (token !== generation || !allowed()) return
      await Promise.all(candidates.slice(i, i + 2).map(tile => move(tile, { ripple: true })))
    }
    scheduleIdle(token)
  }

  function resume() {
    cancel()
    if (allowed()) scheduleIdle(generation)
  }

  document.addEventListener('visibilitychange', resume)
  reducedMotion.addEventListener('change', resume)
  window.addEventListener('resize', () => {
    cancel()
    if (started) {
      build()
      scheduleIdle(generation)
    }
  })
  window.addEventListener('pagehide', cancel)

  return {
    start() {
      cancel()
      started = true
      turn = 0
      build()
      if (allowed()) ripple(generation)
    },
    reset() {
      started = false
      cancel()
      layer?.remove()
    },
  }
}
