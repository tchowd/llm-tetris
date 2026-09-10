import './pages.css'

const posts = [
  { id: 'learning-the-board', category: 'Research', title: 'Teaching a model to see the board.', summary: 'A place to document how board states become examples, decisions, and a learned policy.', sections: [
    ['Start with the state', 'A Tetris board is a compact record of previous decisions. This article template is a place to explain how the board is represented, what the model receives, and what it is asked to predict.'],
    ['Make the experiment readable', 'Add the dataset provenance, a worked example, and the exact model configuration here. Pair each result with the evaluation conditions that produced it.'],
    ['What comes next', 'Close with open questions and the next experiment. This is draft template copy; no experimental results are reported here.'],
  ] },
  { id: 'measuring-a-move', category: 'Research', title: 'What makes a move good?', summary: 'A template for exploring survival, cleared lines, and the difference between a legal action and a useful one.', sections: [
    ['Choose the question', 'Use this section to define the behavior being evaluated. Legal actions, legal top-outs, capped survival, score, and lines answer different questions.'],
    ['Show the evidence', 'Add replayable examples and matched evaluation results here. Keep development observations separate from confirmation results.'],
  ] },
  { id: 'building-tetra', category: 'Engineering', title: 'From pixels to policy.', summary: 'Notes on the systems behind Tetra: the engine, the interface, and the path from a board to a move.', sections: [
    ['One move at a time', 'Describe the path from a board state through the policy and back into the engine. A small annotated example belongs here.'],
    ['Build for replay', 'Document the action format, input serialization, and records needed to reproduce a game. Replace this draft with implementation details as the article develops.'],
  ] },
]

const motif = `<svg viewBox="0 0 240 240" aria-hidden="true"><path d="M24 168h48v48H24zM72 72h48v48h48v48H72zM168 24h48v96h-48z" fill="currentColor"/></svg>`
const arrow = '<span aria-hidden="true">↗</span>'
const header = page => `<header class="section-header">
  <button class="section-brand" data-page="home" aria-label="Back to Tetra landing page">${motif}<span>TETRA</span><span aria-hidden="true">↑</span></button>
  <nav aria-label="Main sections">${['play', 'read', 'data'].map(name => `<button data-page="${name}" ${page === name ? 'aria-current="page"' : ''}>${name}</button>`).join('')}</nav>
  <span class="header-note">A study in every move.</span>
</header>`

function readPage() {
  return `${header('read')}<div class="section-body journal">
    <div class="page-kicker"><span>01 / THE JOURNAL</span><span>Ideas, experiments, observations.</span></div>
    <div class="page-heading"><h1>Field notes<span>.</span></h1><p>Inside the making of Tetra.<br>One question, one experiment, one move at a time.</p></div>
    <div class="template-note"><span class="status-dot"></span> Journal template · Preview articles</div>
    <div class="journal-filters" role="group" aria-label="Filter articles">${['All', 'Research', 'Engineering'].map((label, i) => `<button data-category="${label}" aria-pressed="${i === 0}">${label}<span>${i === 0 ? '03' : i === 1 ? '02' : '01'}</span></button>`).join('')}</div>
    <div class="post-list"></div>
    <footer class="section-footer"><span>THE TETRA JOURNAL</span><span>Build. Observe. Repeat.</span></footer>
  </div>`
}

function postCards(category) {
  return posts.filter(post => category === 'All' || post.category === category).map((post, i) => `<button class="post-card ${i === 0 ? 'featured-post' : ''}" data-post="${post.id}">
    <div class="post-art art-${post.id}">${motif}<span>FIELD NOTE / ${String(posts.indexOf(post) + 1).padStart(2, '0')}</span></div>
    <div class="post-copy"><div class="post-meta"><span>${post.category}</span><span>Draft preview</span></div><h2>${post.title}</h2><p>${post.summary}</p><span class="post-link">Read preview ${arrow}</span></div>
  </button>`).join('')
}

function articlePage(post) {
  return `${header('read')}<article class="section-body article-view">
    <button class="text-button" data-journal>← All field notes</button>
    <div class="post-meta"><span>${post.category}</span><span>Draft preview · Template copy</span></div>
    <h1>${post.title}</h1><p class="article-deck">${post.summary}</p>
    <div class="article-banner">${motif}</div>
    <div class="article-prose">${post.sections.map(([title, copy]) => `<section><h2>${title}</h2><p>${copy}</p></section>`).join('')}</div>
    <footer class="section-footer"><span>TETRA / FIELD NOTES</span><button class="text-button" data-journal>Back to the journal ↑</button></footer>
  </article>`
}

const runs = [
  { model: 'Tetra-1.7B', base: 'Qwen/Qwen3-1.7B', run: 'sft-v1', role: 'Historical baseline' },
  { model: 'Tetra-0.6B', base: 'Qwen/Qwen3-0.6B', run: 'tetra-qwen3-0.6b-combined-v1', role: 'Candidate' },
]

function dataPage() {
  return `${header('data')}<div class="section-body analytics">
    <div class="page-kicker"><span>02 / OBSERVATORY</span><span class="template-note"><span class="status-dot"></span>Template · No results connected</span></div>
    <div class="page-heading"><h1>Every move.<br><span>Measured.</span></h1><p>A place for the evidence.<br>Explore evaluation runs, outcomes, and model behavior.</p></div>
    <div class="data-toolbar"><label>MODEL<select id="model-filter"><option value="all">All models</option>${runs.map(run => `<option>${run.model}</option>`).join('')}</select></label><label>EVALUATION SUITE<select id="suite-filter"><option>Ordinary play</option><option>Long ordinary games</option><option>Recovery</option><option>Stress</option></select></label><button class="outline-button" data-download>CSV template <span aria-hidden="true">↓</span></button></div>
    <div class="metric-grid">${[['Games evaluated', 'Completed game records'], ['Survival', 'Pieces placed per game'], ['Lines cleared', 'Mean lines per game'], ['Illegal actions', 'Share of generated actions']].map(([label, sub]) => `<section class="metric-card"><h2>${label}</h2><strong>—</strong><p>${sub}</p></section>`).join('')}</div>
    <div class="chart-grid"><section class="chart-panel"><div class="panel-heading"><h2>Survival across games</h2><span data-suite-label>Ordinary play</span></div><div class="empty-chart"><div class="chart-gridlines"></div><div class="empty-chart-copy"><span class="empty-chart-icon">↗</span><h3>The picture starts with a run.</h3><p>Game records will populate this chart.<br>No evaluation values are shown yet.</p></div><span class="chart-axis">GAME INDEX</span></div></section>
    <section class="chart-panel"><div class="panel-heading"><h2>How games end</h2><span>OUTCOMES</span></div><div class="outcome-placeholder"><div class="empty-ring"><span>—<small>GAMES</small></span></div><ul>${['Capped survival', 'Legal top-out', 'Illegal action'].map(label => `<li><span>${label}</span><span>—</span></li>`).join('')}</ul></div></section></div>
    <section class="runs-panel"><div class="panel-heading"><h2>Model & run register</h2><span data-run-count>2 MODELS</span></div><div class="table-scroll"><table><thead><tr><th>Model / base</th><th>Run</th><th>Role</th><th>Evaluation results</th></tr></thead><tbody></tbody></table></div><p class="table-note">The candidate has not been promoted. Populate this view with matched evaluations before comparing model performance.</p></section>
    <footer class="section-footer"><span>TETRA / OBSERVATORY</span><span>Measurements before conclusions.</span></footer>
  </div>`
}

function playPage() {
  return `${header('play')}<div class="section-body play-page">
    <div class="page-kicker"><span>00 / THE PLAYGROUND</span><span>Game integration coming next</span></div>
    <div class="play-layout"><div class="play-intro"><span class="template-note"><span class="status-dot"></span>Under construction</span><h1>Your<br>next<br><span>move.</span></h1><p>This is where you’ll play Tetris.<br>The board is reserved. The game comes next.</p><button class="outline-button" data-page="read">Explore the journal ${arrow}</button></div>
    <section class="game-placeholder" aria-label="Future Tetris game placeholder"><div class="game-topline"><span>TETRA / PLAY</span><span>STANDBY</span></div><div class="placeholder-board"><div class="board-piece" aria-hidden="true"></div><div class="board-message"><span>READY FOR WHAT’S NEXT</span><h2>A space to play.</h2><p>Gameplay isn’t connected yet.</p></div></div><div class="game-bottomline"><span>10 COLUMNS</span><span>20 ROWS</span></div></section>
    <aside class="play-status"><div><span>SCORE</span><strong>—</strong></div><div><span>LINES</span><strong>—</strong></div><div><span>LEVEL</span><strong>—</strong></div><p>The live game and controls will be added here.</p></aside></div>
  </div>`
}

export function setupPages(app, landing) {
  const track = document.createElement('div')
  track.className = 'page-track'
  landing.before(track)
  track.append(landing)
  const destination = document.createElement('section')
  destination.className = 'section-page'
  destination.inert = true
  destination.setAttribute('aria-label', 'Tetra sections')
  track.append(destination)
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)')
  let current = 'home', focusTimer
  const renderers = { play: playPage, read: readPage, data: dataPage }

  function updateRuns() {
    const value = destination.querySelector('#model-filter').value
    const selected = runs.filter(run => value === 'all' || run.model === value)
    destination.querySelector('tbody').innerHTML = selected.map(run => `<tr><td><strong>${run.model}</strong><small>${run.base}</small></td><td class="run-id">${run.run}</td><td><span class="role-tag">${run.role}</span></td><td class="results-empty">Not connected</td></tr>`).join('')
    destination.querySelector('[data-run-count]').textContent = `${selected.length} MODEL${selected.length === 1 ? '' : 'S'}`
  }

  function render(page) {
    destination.innerHTML = renderers[page]()
    destination.scrollTop = 0
    if (page === 'read') destination.querySelector('.post-list').innerHTML = postCards('All')
    if (page === 'data') updateRuns()
  }

  function focusPage() {
    if (current === 'home') landing.querySelector('.landing-button')?.focus({ preventScroll: true })
    else {
      const heading = destination.querySelector('h1')
      heading.tabIndex = -1
      heading.focus({ preventScroll: true })
    }
  }

  function navigate(page, push = true) {
    if (page !== 'home' && !renderers[page]) return
    const wasHome = current === 'home'
    current = page
    if (page !== 'home') render(page)
    app.dataset.page = page
    landing.inert = page !== 'home'
    destination.inert = page === 'home'
    landing.dispatchEvent(new Event('tetra:visibility'))
    document.title = page === 'home' ? 'Tetra — Blocks' : `Tetra — ${page[0].toUpperCase() + page.slice(1)}`
    if (push && location.hash !== `#${page}`) history.pushState(null, '', page === 'home' ? `${location.pathname}${location.search}` : `#${page}`)
    clearTimeout(focusTimer)
    focusTimer = setTimeout(focusPage, reducedMotion.matches || (!wasHome && page !== 'home') ? 0 : 900)
  }

  app.addEventListener('tetra:action', event => navigate(event.detail.action))
  destination.addEventListener('click', event => {
    const button = event.target.closest('button')
    if (!button) return
    if (button.dataset.page) navigate(button.dataset.page)
    if (button.dataset.category) {
      destination.querySelectorAll('[data-category]').forEach(item => item.setAttribute('aria-pressed', String(item === button)))
      destination.querySelector('.post-list').innerHTML = postCards(button.dataset.category)
    }
    if (button.dataset.post) {
      const post = posts.find(item => item.id === button.dataset.post)
      destination.innerHTML = articlePage(post)
      destination.scrollTop = 0
      focusPage()
    }
    if (button.hasAttribute('data-journal')) { render('read'); focusPage() }
    if (button.hasAttribute('data-download')) {
      const headers = 'run_id,model,base_model,suite,game_seed,pieces_placed,score,lines,illegal_actions,legal_top_out,capped_survival\n'
      const url = URL.createObjectURL(new Blob([headers], { type: 'text/csv;charset=utf-8' }))
      const link = document.createElement('a')
      link.href = url
      link.download = 'tetra-evaluation-template.csv'
      link.click()
      setTimeout(() => URL.revokeObjectURL(url), 1000)
    }
  })
  destination.addEventListener('change', event => {
    if (event.target.id === 'model-filter') updateRuns()
    if (event.target.id === 'suite-filter') destination.querySelector('[data-suite-label]').textContent = event.target.value
  })
  window.addEventListener('popstate', () => navigate(location.hash.slice(1) || 'home', false))
  if (renderers[location.hash.slice(1)]) navigate(location.hash.slice(1), false)
}
