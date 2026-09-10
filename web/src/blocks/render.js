// All geometry is editable vector data. No reference PNG is loaded by the app.
export const palette = { green: '#abf250', black: '#030202', white: '#fefefe' }
const fill = color => `var(--tetra-${color}, ${palette[color]})`

export const blockSize = { width: 230, height: 327 }

export function normalizeBlocks(blocks) {
  const byId = new Map(blocks.map(block => [block.id, block]))
  // Complete narrow reference crops using their matching full-size motif.
  const fullMotifs = { '15-2': '3', '22': '3', '23': '9' }
  return blocks.map(block => {
    const source = byId.get(fullMotifs[block.id]) ?? block
    const sx = blockSize.width / source.width
    const sy = blockSize.height / source.height
    return {
      ...block, ...blockSize, background: source.background,
      rects: source.rects.map(([x, y, width, height, color]) =>
        [x * sx, y * sy, width * sx, height * sy, color]),
    }
  })
}

export function blockShapes(block) {
  return `<rect width="${block.width}" height="${block.height}" fill="${fill(block.background)}"/>` +
    block.rects.map(([x, y, width, height, color]) =>
      `<rect x="${x}" y="${y}" width="${width}" height="${height}" fill="${fill(color)}"/>`).join('')
}

export function blockSvg(block) {
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${block.width} ${block.height}" shape-rendering="crispEdges" role="img" aria-label="Tetra block ${block.id}: ${block.background} field with geometric shapes">${blockShapes(block)}</svg>`
}

// Every tile occupies one equally sized cell, with aligned row/column seams.
const rows = [
  ['2', '9', '7', '8', '12', '6', '15-2'],
  ['5', '1', '15-1', '3', '4', '20', '23'],
  ['15', '18', '16', '17', '19', '21', '22'],
]
export const composition = rows.flatMap((row, y) => row.map((id, x) =>
  [id, x * blockSize.width, y * blockSize.height, blockSize.width, blockSize.height]))

export function compositionSvg(blocks) {
  blocks = normalizeBlocks(blocks)
  const byId = new Map(blocks.map(block => [block.id, block]))
  const definitions = blocks.map(block => `<symbol id="block-${block.id}" viewBox="0 0 ${block.width} ${block.height}" preserveAspectRatio="none">${blockShapes(block)}</symbol>`).join('')
  const tiles = composition.map(([id, x, y, width, height]) => {
    if (!byId.has(id)) throw new Error(`Missing block ${id}`)
    return `<use href="#block-${id}" x="${x}" y="${y}" width="${width}" height="${height}"/>`
  }).join('')
  const width = rows[0].length * blockSize.width
  const height = rows.length * blockSize.height
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid slice" shape-rendering="crispEdges" role="img" aria-label="Tetra: a mosaic of lime green, black and white pixel blocks"><defs>${definitions}</defs><rect width="${width}" height="${height}" fill="${fill('white')}"/>${tiles}</svg>`
}
