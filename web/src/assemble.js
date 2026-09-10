import { composition, palette } from './blocks/render.js'
import { setupSurroundings } from './surroundings.js'

// A 5 × 7 alphabet keeps the final word made of the same square building blocks.
const letters = {
  T: ['11111', '11111', '00100', '00100', '00100', '00100', '00100'],
  E: ['11111', '10000', '10000', '11110', '10000', '10000', '11111'],
  R: ['11110', '10001', '10001', '11110', '10100', '10010', '10001'],
  A: ['01110', '10001', '10001', '11111', '10001', '10001', '10001'],
}
const ns = 'http://www.w3.org/2000/svg'
const rgb = color => palette[color].slice(1).match(/../g).map(value => parseInt(value, 16))
const ease = t => t * t * t * (t * (t * 6 - 15) + 10)

// Keep contiguous cells together so the movement reads as sliding block shapes.
function mergeRects(rects) {
  const close = (a, b) => Math.abs(a - b) < .01
  const horizontal = []
  for (const rect of [...rects].sort((a, b) => a.y - b.y || a.x - b.x)) {
    const previous = horizontal.find(p => p.color === rect.color && p.tile === rect.tile && close(p.y, rect.y) &&
      close(p.height, rect.height) && close(p.x + p.width, rect.x))
    if (previous) previous.width += rect.width
    else horizontal.push({ ...rect })
  }
  const merged = []
  for (const rect of horizontal.sort((a, b) => a.y - b.y || a.x - b.x)) {
    const previous = merged.find(p => p.color === rect.color && p.tile === rect.tile && close(p.x, rect.x) &&
      close(p.width, rect.width) && close(p.y + p.height, rect.y))
    if (previous) previous.height += rect.height
    else merged.push({ ...rect })
  }
  return merged
}

const wordUnit = (width, height) => Math.min(width * .88 / 29, height * .36 / 7, 52)

function wordPixels(width, height) {
  const unit = wordUnit(width, height)
  const left = (width - 29 * unit) / 2
  const top = (height - 7 * unit) / 2
  return mergeRects([...'TETRA'].flatMap((letter, index) => letters[letter].flatMap((row, y) =>
    [...row].flatMap((cell, x) => cell === '1'
      ? [{ x: left + (index * 6 + x) * unit, y: top + y * unit, width: unit, height: unit }]
      : []))))
}

function sourcePixels(blocks, svg, bounds) {
  const byId = new Map(blocks.map(block => [block.id, block]))
  const matrix = svg.getScreenCTM()
  const result = []
  for (const [id, tileX, tileY] of composition) {
    const block = byId.get(id)
    // Partition only at original contours, preserving the large sliding shapes.
    const edges = (size, axis) => [...new Set([0, size, ...block.rects.flatMap(rect =>
      [rect[axis], rect[axis] + rect[axis + 2]])])].sort((a, b) => a - b)
    const xs = edges(block.width, 0), ys = edges(block.height, 1)
    for (let yi = 0; yi < ys.length - 1; yi++) {
      for (let xi = 0; xi < xs.length - 1; xi++) {
        const x = xs[xi], y = ys[yi], w = xs[xi + 1] - x, h = ys[yi + 1] - y
        if (w < .1 || h < .1) continue
        let color = block.background
        for (const [rx, ry, rw, rh, fill] of block.rects) {
          if (x + w / 2 >= rx && x + w / 2 < rx + rw && y + h / 2 >= ry && y + h / 2 < ry + rh) color = fill
        }
        if (color === block.background) continue
        const point = new DOMPoint(tileX + x, tileY + y).matrixTransform(matrix)
        const pw = w * matrix.a, ph = h * matrix.d
        const left = Math.max(point.x, bounds.x), top = Math.max(point.y, bounds.y)
        const right = Math.min(point.x + pw, bounds.x + bounds.width)
        const bottom = Math.min(point.y + ph, bounds.y + bounds.height)
        if (right <= left || bottom <= top) continue
        result.push({ x: left, y: top, width: right - left, height: bottom - top, color, tile: id })
      }
    }
  }
  return mergeRects(result).sort((a, b) => a.x - b.x || a.y - b.y)
}

export function setupAssembly(container, blocks) {
  const artwork = container.querySelector('svg')
  const button = document.createElement('button')
  button.className = 'start-button'
  button.type = 'button'
  button.innerHTML = '<span>Start</span><span aria-hidden="true">↗</span>'
  container.append(button)
  const status = document.createElement('span')
  status.className = 'sr-only'
  status.setAttribute('role', 'status')
  container.append(status)
  let state = 'idle', overlay, particles = [], frame, startTime
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)')

  function layout() {
    const strokes = wordPixels(innerWidth, innerHeight)
    const unit = wordUnit(innerWidth, innerHeight)
    const bounds = {
      x: (innerWidth - 31 * unit) / 2, y: (innerHeight - 9 * unit) / 2,
      width: 31 * unit, height: 9 * unit,
    }
    const matrix = artwork.getScreenCTM()
    const byId = new Map(blocks.map(block => [block.id, block]))
    const tiles = composition.map(([id, x, y, width, height]) => {
      const point = new DOMPoint(x, y).matrixTransform(matrix)
      const block = byId.get(id)
      // Keep the tile's own field; choose a contrasting foreground from the palette.
      const color = block.background === 'black' ? 'white'
        : block.background === 'green' ? 'black'
        : block.rects.some(rect => rect[4] === 'green') ? 'green' : 'black'
      return { id, x: point.x, y: point.y, width: width * matrix.a,
        height: height * matrix.d, background: block.background, color }
    })
    // A letter can cross a tile seam, changing ink color exactly at that seam.
    const targets = strokes.flatMap(stroke => tiles.flatMap(tile => {
      const x = Math.max(stroke.x, tile.x), y = Math.max(stroke.y, tile.y)
      const right = Math.min(stroke.x + stroke.width, tile.x + tile.width)
      const bottom = Math.min(stroke.y + stroke.height, tile.y + tile.height)
      return right > x && bottom > y
        ? [{ x, y, width: right - x, height: bottom - y, color: tile.color, tile: tile.id }]
        : []
    }))
    return { targets, tiles, bounds }
  }

  const surroundings = setupSurroundings(container, blocks, layout, reducedMotion)

  function stage(bounds, tiles) {
    overlay.setAttribute('viewBox', `0 0 ${innerWidth} ${innerHeight}`)
    overlay.setAttribute('shape-rendering', 'crispEdges')
    overlay.innerHTML = `<defs><clipPath id="assembly-window"><rect x="${bounds.x}" y="${bounds.y}" width="${bounds.width}" height="${bounds.height}"/></clipPath></defs>
      <g clip-path="url(#assembly-window)"><g class="tile-backgrounds">${tiles.map(tile =>
        `<rect data-tile="${tile.id}" x="${tile.x}" y="${tile.y}" width="${tile.width}" height="${tile.height}" fill="${palette[tile.background]}"/>`).join('')}</g><g class="sliding-shapes"></g></g>`
    return overlay.querySelector('.sliding-shapes')
  }

  function drawFinal() {
    const { targets, bounds, tiles } = layout()
    const group = stage(bounds, tiles)
    group.innerHTML = targets.map(p => `<rect data-tile="${p.tile}" x="${p.x}" y="${p.y}" width="${p.width}" height="${p.height}" fill="${palette[p.color]}"/>`).join('')
  }

  function finish() {
    cancelAnimationFrame(frame)
    state = 'complete'
    container.dataset.state = state
    drawFinal()
    overlay.setAttribute('role', 'img')
    overlay.setAttribute('aria-label', 'TETRA')
    overlay.removeAttribute('aria-hidden')
    button.disabled = false
    button.innerHTML = '<span>Replay</span><span aria-hidden="true">↺</span>'
    status.textContent = 'TETRA assembled.'
    particles = []
  }

  function tick(now) {
    const elapsed = now - startTime
    for (const particle of particles) {
      const { source, target, element, delay, horizontalFirst } = particle
      const progress = Math.max(0, Math.min(1, (elapsed - delay) / 1550))
      // Slide along one axis, then the other: no rotation or airborne motion.
      const first = ease(Math.min(1, progress * 2))
      const second = ease(Math.max(0, progress * 2 - 1))
      const tx = horizontalFirst ? first : second
      const ty = horizontalFirst ? second : first
      element.setAttribute('x', source.x + (target.x - source.x) * tx)
      element.setAttribute('y', source.y + (target.y - source.y) * ty)
      element.setAttribute('width', source.width + (target.width - source.width) * ease(progress))
      element.setAttribute('height', source.height + (target.height - source.height) * ease(progress))
      if (source.color !== target.color) {
        const from = rgb(source.color), to = rgb(target.color), t = ease(progress)
        element.setAttribute('fill', `rgb(${from.map((value, i) => Math.round(value + (to[i] - value) * t)).join(',')})`)
      }
    }
    if (elapsed >= 2000) finish()
    else frame = requestAnimationFrame(tick)
  }

  function start() {
    const { targets, bounds, tiles } = layout()
    const sources = sourcePixels(blocks, artwork, bounds)
    overlay = document.createElementNS(ns, 'svg')
    overlay.classList.add('assembly')
    overlay.setAttribute('aria-hidden', 'true')
    container.append(overlay)
    const group = stage(bounds, tiles)
    const assignments = []
    for (const tile of tiles) {
      const tileTargets = targets.filter(target => target.tile === tile.id)
      const tileSources = sources.filter(source => source.tile === tile.id)
      // Divide a source only when a tile needs more pieces to make its strokes.
      while (tileSources.length && tileSources.length < tileTargets.length) {
        const largest = [...tileSources].sort((a, b) => b.width * b.height - a.width * a.height)[0]
        if (largest.width >= largest.height) {
          largest.width /= 2
          tileSources.push({ ...largest, x: largest.x + largest.width })
        } else {
          largest.height /= 2
          tileSources.push({ ...largest, y: largest.y + largest.height })
        }
      }
      if (!tileSources.length && tileTargets.length) {
        tileTargets.forEach((target, i) => assignments.push({
          source: { ...target, y: i % 2 ? bounds.y - target.height : bounds.y + bounds.height }, target,
        }))
      }
      tileSources.sort((a, b) => a.x - b.x || a.y - b.y)
      tileSources.forEach((source, index) => {
        const target = tileTargets.length
          ? tileTargets[Math.floor(index * tileTargets.length / tileSources.length)]
          : { ...source, y: bounds.y + bounds.height, color: source.color }
        assignments.push({ source, target })
      })
    }
    particles = assignments.map(({ source, target }, index) => {
      const element = document.createElementNS(ns, 'rect')
      for (const attr of ['x', 'y', 'width', 'height']) element.setAttribute(attr, source[attr])
      element.setAttribute('fill', palette[source.color])
      group.append(element)
      return { source, target, element, delay: index * 37 % 360, horizontalFirst: index % 2 === 0 }
    })
    surroundings.start()
    // Establish the original palette on the newly inserted layers before changing
    // state, so their gray transition starts alongside the letter movement.
    for (const layer of [artwork, container.querySelector('.surroundings'), overlay.querySelector('.tile-backgrounds')]) {
      if (layer) getComputedStyle(layer).filter
    }
    state = 'animating'
    container.dataset.state = state
    button.disabled = true
    button.innerHTML = '<span>Assembling</span><span aria-hidden="true">·</span>'
    status.textContent = 'Sliding shapes into TETRA.'
    if (reducedMotion.matches) return finish()
    startTime = performance.now()
    frame = requestAnimationFrame(tick)
  }

  button.addEventListener('click', () => {
    if (state === 'animating') return
    if (state === 'complete') {
      surroundings.reset()
      overlay.remove()
      state = 'idle'
      container.dataset.state = state
      button.innerHTML = '<span>Start</span><span aria-hidden="true">↗</span>'
      status.textContent = ''
      // Show the original arrangement briefly before replaying the transformation.
      button.disabled = true
      setTimeout(start, reducedMotion.matches ? 0 : 350)
    } else start()
  })
  window.addEventListener('resize', () => {
    if (state === 'animating') finish()
    else if (state === 'complete') drawFinal()
  })
}
