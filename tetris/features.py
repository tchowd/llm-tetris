"""Pure board-array feature formulas. Stage 1's prompt features (heights,
holes, wells, bumpiness) and Stage 2's Dellacherie `holes_total` / well-sums
both read these, so there is exactly one definition of each quantity."""
from __future__ import annotations

from .board import HEIGHT, WIDTH


def column_heights(board: list[list[str]]) -> list[int]:
    heights = [0] * WIDTH
    for col in range(WIDTH):
        for row in range(HEIGHT):
            if board[row][col] != ".":
                heights[col] = HEIGHT - row
                break
    return heights


def column_holes(board: list[list[str]]) -> list[int]:
    holes = [0] * WIDTH
    for col in range(WIDTH):
        seen_filled = False
        for row in range(HEIGHT):
            if board[row][col] != ".":
                seen_filled = True
            elif seen_filled:
                holes[col] += 1
    return holes


def wells(heights: list[int]) -> list[int]:
    result = [0] * WIDTH
    for col in range(WIDTH):
        left = heights[col - 1] if col > 0 else HEIGHT
        right = heights[col + 1] if col < WIDTH - 1 else HEIGHT
        if heights[col] < left and heights[col] < right:
            result[col] = min(left, right) - heights[col]
    return result


def bumpiness(heights: list[int]) -> int:
    return sum(abs(heights[i] - heights[i + 1]) for i in range(WIDTH - 1))
