"""Convert the supplied orthogonal artwork into editable rectangle data and SVGs.

Run with Python + Pillow + NumPy. This is an offline authoring tool; the website
has no Python dependency and never loads the PNG references at runtime.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PALETTE = {"black": "#030202", "white": "#fefefe", "green": "#abf250"}
COLORS = list(PALETTE)


def classify(pixels):
    pixels = pixels.astype(int)
    green = (pixels[:, :, 1] > 150) & (pixels[:, :, 0] < 220) & (pixels[:, :, 2] < 150)
    return np.where(green, 2, np.where(pixels.mean(axis=2) > 128, 1, 0))


def boundaries(labels, axis):
    # Long edges distinguish the artwork from compression noise and paper grain.
    counts = np.count_nonzero(np.diff(labels, axis=axis), axis=1 - axis)
    candidates = np.flatnonzero(counts >= 12) + 1
    groups = []
    for point in candidates:
        if groups and point - groups[-1][-1] <= 3:
            groups[-1].append(int(point))
        else:
            groups.append([int(point)])
    edges = [max(group, key=lambda p: counts[p - 1]) for group in groups]
    return [0] + [edge for edge in edges if 3 < edge < labels.shape[axis] - 3] + [labels.shape[axis]]


def trace(path):
    pixels = np.asarray(Image.open(path).convert("RGB"))
    labels = classify(pixels)
    height, width = labels.shape
    xs, ys = boundaries(labels, 1), boundaries(labels, 0)
    background = int(np.bincount(labels.ravel(), minlength=3).argmax())
    rectangles = []
    active = {}
    for top, bottom in zip(ys, ys[1:]):
        runs = []
        for left, right in zip(xs, xs[1:]):
            color = int(np.bincount(labels[top:bottom, left:right].ravel(), minlength=3).argmax())
            if color == background:
                continue
            if runs and runs[-1][0] + runs[-1][2] == left and runs[-1][4] == color:
                runs[-1][2] += right - left
            else:
                runs.append([left, top, right - left, bottom - top, color])
        next_active = {}
        for rect in runs:
            key = (rect[0], rect[2], rect[4])
            if key in active:
                index = active[key]
                rectangles[index][3] += bottom - top
            else:
                index = len(rectangles)
                rectangles.append(rect)
            next_active[key] = index
        active = next_active
    reconstructed = np.full_like(labels, background)
    for x, y, w, h, color in rectangles:
        reconstructed[y:y+h, x:x+w] = color
    agreement = float(np.mean(labels == reconstructed))
    if agreement < .99:
        raise ValueError(f"Check geometry for {path.name}: {agreement:.2%}")
    return {
        "id": path.stem.split(" ")[-1], "source": path.name,
        "width": width, "height": height, "background": COLORS[background],
        "rects": [[x, y, w, h, COLORS[color]] for x, y, w, h, color in rectangles],
    }, agreement


def svg(block):
    shapes = [f'<rect width="{block["width"]}" height="{block["height"]}" fill="{PALETTE[block["background"]]}"/>']
    for x, y, w, h, color in block["rects"]:
        shapes.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{PALETTE[color]}"/>')
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {block["width"]} {block["height"]}">\n  <title>Tetra block {block["id"]}</title>\n  ' + '\n  '.join(shapes) + '\n</svg>\n'


if __name__ == "__main__":
    sources = sorted((ROOT / "assets").glob("hf_*.png"), key=lambda p: tuple(int(n) for n in p.stem.split(" ")[-1].split("-")))
    catalog = []
    output = ROOT / "assets/blocks"
    output.mkdir(exist_ok=True)
    for path in sources:
        block, agreement = trace(path)
        catalog.append(block)
        (output / f'block-{block["id"]}.svg').write_text(svg(block))
        print(f'Block {block["id"]:>4}: {len(block["rects"]):2} rectangles, {agreement:.2%} color-region agreement')
    (ROOT / "web/src/blocks/catalog.json").write_text(json.dumps(catalog, indent=2) + "\n")
    subprocess.run(["node", str(ROOT / "web/scripts/export-composition.js")], check=True)
