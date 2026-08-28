// Thin client: all game logic lives server-side in tetris.engine.Game.
// This file only renders snapshots and turns key presses into
// POST /api/games/{id}/step calls.
import "./style.css";

const WIDTH = 10;
const HEIGHT = 20;

// Cosmetic-only spawn-orientation shapes for the "next piece" preview box.
// Not used for any game logic; the ghost on the main board comes straight
// from the server's `legal` cells.
const PREVIEW_SHAPES = {
  I: [[1, 0], [1, 1], [1, 2], [1, 3]],
  O: [[0, 0], [0, 1], [1, 0], [1, 1]],
  T: [[0, 1], [1, 0], [1, 1], [1, 2]],
  S: [[0, 1], [0, 2], [1, 0], [1, 1]],
  Z: [[0, 0], [0, 1], [1, 1], [1, 2]],
  J: [[0, 0], [1, 0], [1, 1], [1, 2]],
  L: [[0, 2], [1, 0], [1, 1], [1, 2]],
};

const boardEl = document.getElementById("board");
const gameOverEl = document.getElementById("game-over");
const nextGridEl = document.getElementById("next-grid");
const promptEl = document.getElementById("prompt-text");
const statScore = document.getElementById("stat-score");
const statLines = document.getElementById("stat-lines");
const statTurn = document.getElementById("stat-turn");
const statSeed = document.getElementById("stat-seed");
const newGameForm = document.getElementById("new-game-form");
const seedInput = document.getElementById("seed-input");
const teacherStepBtn = document.getElementById("teacher-step-btn");
const teacherAutoplayBtn = document.getElementById("teacher-autoplay-btn");
const teacherActionEl = document.getElementById("teacher-action");

let gameId = null;
let snapshot = null;
let selectedRot = 0;
let selectedX = 0;
let stepInFlight = false;
let autoplayTimer = null;

const toastEl = document.getElementById("toast");
let toastTimer = null;

function showToast(message) {
  toastEl.textContent = message;
  toastEl.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.add("hidden"), 2000);
}

function buildCellGrid(container, width, height, cellClass) {
  container.innerHTML = "";
  const cells = [];
  for (let r = 0; r < height; r++) {
    const row = [];
    for (let c = 0; c < width; c++) {
      const cell = document.createElement("div");
      cell.className = cellClass;
      container.appendChild(cell);
      row.push(cell);
    }
    cells.push(row);
  }
  return cells;
}

const boardCells = buildCellGrid(boardEl, WIDTH, HEIGHT, "cell");
const nextCells = buildCellGrid(nextGridEl, 4, 4, "cell");

function legalGroupedByRot(legal) {
  const groups = new Map();
  for (const p of legal) {
    if (!groups.has(p.rot)) groups.set(p.rot, []);
    groups.get(p.rot).push(p);
  }
  for (const list of groups.values()) list.sort((a, b) => a.x - b.x);
  return groups;
}

function clampSelection() {
  if (!snapshot) return;
  const groups = legalGroupedByRot(snapshot.legal);
  const rots = [...groups.keys()].sort((a, b) => a - b);
  if (rots.length === 0) return; // game over, nothing legal

  if (!rots.includes(selectedRot)) {
    selectedRot = rots.reduce((closest, r) =>
      Math.abs(r - selectedRot) < Math.abs(closest - selectedRot) ? r : closest
    , rots[0]);
  }
  const xs = groups.get(selectedRot).map((p) => p.x);
  if (!xs.includes(selectedX)) {
    selectedX = xs.reduce((closest, x) =>
      Math.abs(x - selectedX) < Math.abs(closest - selectedX) ? x : closest
    , xs[0]);
  }
}

function currentSelectionPlacement() {
  if (!snapshot) return null;
  return snapshot.legal.find((p) => p.rot === selectedRot && p.x === selectedX) || null;
}

function render() {
  if (!snapshot) return;

  for (let r = 0; r < HEIGHT; r++) {
    const rowStr = snapshot.board[r];
    for (let c = 0; c < WIDTH; c++) {
      const ch = rowStr[c];
      boardCells[r][c].className = "cell" + (ch !== "." ? ` ${ch}` : "");
    }
  }

  const ghost = currentSelectionPlacement();
  if (ghost && !snapshot.game_over) {
    for (const [r, c] of ghost.cells) {
      if (boardCells[r][c].className === "cell") {
        boardCells[r][c].className = "cell ghost";
      }
    }
  }

  const nextShape = PREVIEW_SHAPES[snapshot.next] || [];
  for (let r = 0; r < 4; r++) {
    for (let c = 0; c < 4; c++) {
      nextCells[r][c].className = "cell";
    }
  }
  for (const [r, c] of nextShape) {
    nextCells[r][c].className = `cell ${snapshot.next}`;
  }

  promptEl.textContent = snapshot.prompt;
  statScore.textContent = snapshot.score;
  statLines.textContent = snapshot.lines;
  statTurn.textContent = snapshot.turn;
  statSeed.textContent = snapshot.seed;

  gameOverEl.classList.toggle("hidden", !snapshot.game_over);

  teacherStepBtn.disabled = snapshot.game_over;
  teacherAutoplayBtn.disabled = snapshot.game_over;
  if (snapshot.game_over) stopAutoplay();
}

function stopAutoplay() {
  clearInterval(autoplayTimer);
  autoplayTimer = null;
  teacherAutoplayBtn.textContent = "Auto-play";
  teacherAutoplayBtn.classList.remove("active");
}

async function newGame(seed) {
  if (stepInFlight) return;
  stopAutoplay();
  stepInFlight = true;
  try {
    const body = seed === null || seed === undefined || seed === "" ? {} : { seed: Number(seed) };
    const res = await fetch("/api/games", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      showToast("could not start a new game");
      return;
    }
    snapshot = await res.json();
    gameId = snapshot.game_id;
    selectedRot = 0;
    selectedX = 0;
    teacherActionEl.textContent = " ";
    clampSelection();
    render();
  } finally {
    stepInFlight = false;
  }
}

async function teacherStep() {
  if (!gameId || !snapshot || snapshot.game_over || stepInFlight) return;
  stepInFlight = true;
  try {
    const res = await fetch(`/api/games/${gameId}/teacher-step`, { method: "POST" });
    if (!res.ok) {
      const body = await res.json().catch(() => null);
      showToast(body?.detail || "teacher move failed");
      stopAutoplay();
      return;
    }
    snapshot = await res.json();
    const { rot, x } = snapshot.teacher_action;
    teacherActionEl.textContent = `teacher played rot=${rot} x=${x}`;
    clampSelection();
    render();
  } finally {
    stepInFlight = false;
  }
}

function toggleAutoplay() {
  if (autoplayTimer) {
    stopAutoplay();
    return;
  }
  teacherAutoplayBtn.textContent = "Stop";
  teacherAutoplayBtn.classList.add("active");
  autoplayTimer = setInterval(() => {
    if (!snapshot || snapshot.game_over) {
      stopAutoplay();
      return;
    }
    teacherStep();
  }, 400);
}

async function step(rot, x) {
  if (!gameId || !snapshot || snapshot.game_over || stepInFlight) return;
  stepInFlight = true;
  try {
    const res = await fetch(`/api/games/${gameId}/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ rot, x }),
    });
    if (!res.ok) {
      // Should only happen if the selection went stale mid-request; the
      // board itself is unaffected since the server rejected the move.
      const body = await res.json().catch(() => null);
      showToast(body?.detail || "move rejected");
      return;
    }
    snapshot = await res.json();
    clampSelection();
    render();
  } finally {
    stepInFlight = false;
  }
}

function moveSelection(dx) {
  if (!snapshot) return;
  const groups = legalGroupedByRot(snapshot.legal);
  const xs = (groups.get(selectedRot) || []).map((p) => p.x);
  if (xs.length === 0) return;
  const idx = xs.indexOf(selectedX);
  const nextIdx = Math.min(Math.max(idx + dx, 0), xs.length - 1);
  selectedX = xs[nextIdx];
  render();
}

function rotateSelection(direction) {
  if (!snapshot) return;
  const groups = legalGroupedByRot(snapshot.legal);
  const rots = [...groups.keys()].sort((a, b) => a - b);
  if (rots.length === 0) return;
  const idx = rots.indexOf(selectedRot);
  const nextIdx = (idx + direction + rots.length) % rots.length;
  selectedRot = rots[nextIdx];
  const xs = groups.get(selectedRot).map((p) => p.x);
  if (!xs.includes(selectedX)) {
    selectedX = xs.reduce((closest, x) =>
      Math.abs(x - selectedX) < Math.abs(closest - selectedX) ? x : closest
    , xs[0]);
  }
  render();
}

document.addEventListener("keydown", (e) => {
  if (document.activeElement === seedInput) return;
  switch (e.code) {
    case "ArrowLeft":
      e.preventDefault();
      moveSelection(-1);
      break;
    case "ArrowRight":
      e.preventDefault();
      moveSelection(1);
      break;
    case "ArrowUp":
      e.preventDefault();
      rotateSelection(1);
      break;
    case "KeyZ":
      e.preventDefault();
      rotateSelection(-1);
      break;
    case "Space":
    case "Enter":
      e.preventDefault();
      step(selectedRot, selectedX);
      break;
  }
});

newGameForm.addEventListener("submit", (e) => {
  e.preventDefault();
  newGame(seedInput.value);
  seedInput.value = "";
  seedInput.blur();
});

teacherStepBtn.addEventListener("click", () => teacherStep());
teacherAutoplayBtn.addEventListener("click", () => toggleAutoplay());

newGame(null);
