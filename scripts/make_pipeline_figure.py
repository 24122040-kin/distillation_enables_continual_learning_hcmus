"""Draw the SDFT training pipeline (slide 9) as presentation/assets/pipeline_sdft.png."""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
STUDENT, TEACHER, DATA, LOSS = "#2a78d6", "#1baf7a", "#8f8d85", "#eb6834"
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans"], "figure.facecolor": SURFACE})


def box(ax, x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                facecolor=color + "1f", edgecolor=color, linewidth=2))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=15, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=11.5, color=INK2)


def arrow(ax, a, b, color=INK2, text=None, off=(0, 0.13), rad=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=18, color=color, linewidth=1.8,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", va="bottom",
                fontsize=11, color=INK2)


def main(out: str) -> None:
    fig, ax = plt.subplots(figsize=(13, 6.6))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6.6)
    ax.axis("off")
    box(ax, 0.1, 2.3, 2.6, 1.5, "Câu hỏi x", "+ lời giải mẫu c\n(chỉ Teacher xem)", DATA)
    box(ax, 3.6, 3.9, 3.0, 1.4, "Student  πθ(· | x)", "LoRA đang học, chỉ thấy x", STUDENT)
    box(ax, 3.6, 0.7, 3.0, 1.4, "Teacher  πφ(· | x, c)", "EMA của Student, thấy cả c", TEACHER)
    box(ax, 7.5, 3.9, 2.4, 1.4, "Rollout y ~ πθ", "1 mẫu / câu (on-policy)", STUDENT)
    box(ax, 10.2, 2.2, 2.9, 1.6, "KL từng token", "Σ_t KL(πφ ‖ πθ) trên y\n(forward, giải tích)", LOSS)
    arrow(ax, (2.7, 3.35), (3.6, 4.5), text="x", off=(-0.15, 0.05))
    arrow(ax, (2.7, 2.75), (3.6, 1.5), text="x, c", off=(-0.25, -0.35))
    arrow(ax, (6.6, 4.6), (7.5, 4.6), text="sinh")
    arrow(ax, (9.9, 4.3), (10.9, 3.8), text="y", off=(0.1, 0.0))
    arrow(ax, (8.2, 3.9), (6.6, 1.8), text="chấm lại y", off=(0.75, -0.25))
    arrow(ax, (6.6, 1.1), (10.6, 2.2), text="logits Teacher", off=(0.5, -0.55))
    arrow(ax, (12.0, 3.8), (5.6, 5.3), color=LOSS, rad=0.42)
    ax.text(9.6, 6.05, "∇θ: cập nhật Student (chỉ Student nhận gradient)", ha="center", fontsize=11.5, color=LOSS)
    arrow(ax, (4.6, 3.9), (4.6, 2.1), color=TEACHER)
    ax.text(4.45, 3.0, "EMA\nφ ← (1−α)φ + αθ", ha="right", va="center", fontsize=11, color=TEACHER)
    ax.text(0.1, 6.3, "SDFT: tự chưng cất on-policy, không cần hàm thưởng", fontsize=17, fontweight="bold", color=INK)
    ax.text(0.1, 0.15, "Che 3 token đầu khi tính KL;  α = 0.01;  Teacher và Student dùng chung mô hình gốc, khác adapter LoRA.",
            fontsize=11, color=INK2)
    fig.savefig(os.path.join(out, "pipeline_sdft.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "presentation/assets")
