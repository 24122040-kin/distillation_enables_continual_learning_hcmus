"""Render the method formulas (slides 8, 11, 13) as transparent PNGs for the Canva deck.

Usage::

    python scripts/make_formula_images.py presentation/assets/formulas

White text on a transparent background (the deck uses a dark blue theme), 300 dpi.
"""

import io
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "serif"})

PT = r"\pi_\theta"
PF = r"\pi_\phi"
E = r"\mathbb{E}"

FORMULAS = {
    # Slide 8: training objective
    "s08_objective": [
        rf"$\min_\theta\ \mathcal{{L}}(\theta) = {E}_{{(x,c)\sim\mathcal{{D}}}}\left[ D_{{KL}}\left( {PT}(\cdot\,|\,x)\ \|\ {PF}(\cdot\,|\,x,c) \right) \right]$",
        rf"$\qquad\quad = {E}_{{(x,c)\sim\mathcal{{D}},\ y\sim {PT}(\cdot|x)}}\left[ \log \frac{{{PT}(y\,|\,x)}}{{{PF}(y\,|\,x,c)}} \right]$",
    ],
    # Slide 11: reverse-KL gradient (theory) and forward-KL loss (implementation)
    "s11_reverse_kl": [
        rf"$\nabla_\theta \mathcal{{L}}_{{rev}} = {E}_{{y\sim {PT}}}\left[ \sum_t \, \sum_{{v\in V}} {PT}(v\,|\,y_{{<t}},x)\ \log\frac{{{PT}(v\,|\,y_{{<t}},x)}}{{{PF}(v\,|\,y_{{<t}},x,c)}}\ \nabla_\theta \log {PT}(v\,|\,y_{{<t}},x) \right]$",
    ],
    "s11_forward_kl": [
        rf"$\mathcal{{L}}_{{fwd}} = \sum_t \, \sum_{{v\in V}} {PF}(v\,|\,y_{{<t}},x,c)\ \log\frac{{{PF}(v\,|\,y_{{<t}},x,c)}}{{{PT}(v\,|\,y_{{<t}},x)}}$",
    ],
    # Slide 13: five-step derivation
    "s13_step1": [rf"$\pi_{{k+1}} = \mathrm{{arg\,max}}_\pi\ {E}_{{y\sim\pi}}\left[ r(y,x) \right] - \beta\, D_{{KL}}\left( \pi(\cdot\,|\,x)\ \|\ \pi_k(\cdot\,|\,x) \right)$"],
    "s13_step2": [rf"$\pi^*_{{k+1}}(y\,|\,x) \propto \pi_k(y\,|\,x)\ \exp\left( r(y,x)/\beta \right)$"],
    "s13_step3": [rf"$r(y,x) = \log \pi^*_{{k+1}}(y\,|\,x) - \log \pi_k(y\,|\,x)$"],
    "s13_step4": [
        rf"$\pi^*_{{k+1}}(y\,|\,x) \approx \pi(y\,|\,x,c)$",
        rf"$\Rightarrow\ r_t = \log \pi(y_t\,|\,y_{{<t}},x,c) - \log \pi_k(y_t\,|\,y_{{<t}},x), \qquad r(y,x,c) = \sum_t r_t$",
    ],
    "s13_step5": [rf"$\nabla_\theta J = {E}_{{y\sim\pi_k}}\left[ \log\frac{{\pi(y\,|\,x,c)}}{{\pi_k(y\,|\,x)}}\ \nabla_\theta \log \pi_k(y\,|\,x) \right]$"],
}


def _line(text, size, color):
    """One formula line as an RGBA array (tight bounding box)."""
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, text, fontsize=size, color=color)
    buf = io.BytesIO()
    fig.savefig(buf, dpi=300, transparent=True, bbox_inches="tight", pad_inches=0.04, format="png")
    plt.close(fig)
    buf.seek(0)
    return plt.imread(buf)


def render(lines, path, size=26, color="white", gap_px=40):
    """Stack the lines left-aligned with a fixed vertical gap and save one PNG."""
    imgs = [_line(t, size, color) for t in lines]
    w = max(i.shape[1] for i in imgs)
    h = sum(i.shape[0] for i in imgs) + gap_px * (len(imgs) - 1)
    out = np.zeros((h, w, 4), dtype=np.float32)
    y = 0
    for i in imgs:
        out[y:y + i.shape[0], :i.shape[1]] = i
        y += i.shape[0] + gap_px
    plt.imsave(path, out)


def main(out: str) -> None:
    os.makedirs(out, exist_ok=True)
    for name, lines in FORMULAS.items():
        render(lines, os.path.join(out, f"{name}.png"))
    print("wrote", sorted(os.listdir(out)))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "presentation/assets/formulas")
