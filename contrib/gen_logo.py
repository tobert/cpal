#!/usr/bin/env python3
"""Generate the cpal logo as SVG.

Concept: A warm coral circle containing a white magnifying glass
whose handle curves into a protective "c" shape. The lens has a
subtle code bracket inside — a helper that examines code carefully
and keeps you out of trouble.
"""

import math

# Anthropic-adjacent warm palette
CORAL = "#D4714E"       # warm terracotta — primary
CORAL_DARK = "#B85A3A"  # darker shade for depth
WHITE = "#FFFFFF"
CREAM = "#FFF8F4"       # warm off-white

# Canvas
SIZE = 512
CX, CY = SIZE / 2, SIZE / 2
R = SIZE * 0.44  # outer circle radius


def circle(cx, cy, r, **attrs):
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" {a}/>'


def path(d, **attrs):
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<path d="{d}" {a}/>'


def svg_doc(width, height, *children):
    inner = "\n  ".join(children)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  {inner}
</svg>"""


def make_logo():
    elements = []

    # Background circle — warm coral
    elements.append(circle(CX, CY, R, fill=CORAL))

    # Subtle inner shadow ring for depth
    elements.append(circle(CX, CY, R * 0.92, fill="none",
                           stroke=CORAL_DARK, stroke_width="2", opacity="0.3"))

    # === Magnifying glass ===
    # The lens (circle) is positioned upper-left of center
    lens_cx = CX - SIZE * 0.04
    lens_cy = CY - SIZE * 0.06
    lens_r = SIZE * 0.155

    # Lens ring (thick white stroke)
    elements.append(circle(lens_cx, lens_cy, lens_r,
                           fill="none", stroke=WHITE,
                           stroke_width=f"{SIZE * 0.04:.1f}",
                           stroke_linecap="round"))

    # Lens interior — subtle tinted glass effect
    elements.append(circle(lens_cx, lens_cy, lens_r * 0.75,
                           fill=WHITE, opacity="0.12"))

    # Handle — curves down-right from the lens, forming a protective "c" curve
    # Start point: where handle meets lens ring (bottom-right of lens)
    angle_start = math.radians(40)  # ~5 o'clock on the lens
    hx1 = lens_cx + (lens_r + SIZE * 0.015) * math.cos(angle_start)
    hy1 = lens_cy + (lens_r + SIZE * 0.015) * math.sin(angle_start)

    # Handle end point — curves down then sweeps right
    hx2 = CX + SIZE * 0.20
    hy2 = CY + SIZE * 0.25

    # Control points for a nice protective curve
    cp1x = hx1 + SIZE * 0.06
    cp1y = hy1 + SIZE * 0.10
    cp2x = hx2 - SIZE * 0.04
    cp2y = hy2 - SIZE * 0.06

    handle_d = (
        f"M {hx1:.1f} {hy1:.1f} "
        f"C {cp1x:.1f} {cp1y:.1f}, {cp2x:.1f} {cp2y:.1f}, {hx2:.1f} {hy2:.1f}"
    )
    elements.append(path(handle_d, fill="none", stroke=WHITE,
                         stroke_width=f"{SIZE * 0.04:.1f}",
                         stroke_linecap="round"))

    # === Code angle brackets inside the lens: < > ===
    # These say "I look at code"
    bracket_size = lens_r * 0.38
    bracket_y = lens_cy
    bracket_gap = lens_r * 0.15  # gap between < and >

    # Left bracket <
    lbx = lens_cx - bracket_gap
    lb_d = (
        f"M {lbx:.1f} {bracket_y - bracket_size:.1f} "
        f"L {lbx - bracket_size * 0.6:.1f} {bracket_y:.1f} "
        f"L {lbx:.1f} {bracket_y + bracket_size:.1f}"
    )
    elements.append(path(lb_d, fill="none", stroke=WHITE,
                         stroke_width=f"{SIZE * 0.018:.1f}",
                         stroke_linecap="round", stroke_linejoin="round"))

    # Right bracket >
    rbx = lens_cx + bracket_gap
    rb_d = (
        f"M {rbx:.1f} {bracket_y - bracket_size:.1f} "
        f"L {rbx + bracket_size * 0.6:.1f} {bracket_y:.1f} "
        f"L {rbx:.1f} {bracket_y + bracket_size:.1f}"
    )
    elements.append(path(rb_d, fill="none", stroke=WHITE,
                         stroke_width=f"{SIZE * 0.018:.1f}",
                         stroke_linecap="round", stroke_linejoin="round"))

    # === Small sparkle/star near the lens — insight ===
    # A tiny 4-point star at upper-right of the lens
    sx = lens_cx + lens_r * 0.75
    sy = lens_cy - lens_r * 0.65
    ss = SIZE * 0.025  # sparkle size
    sparkle_d = (
        f"M {sx:.1f} {sy - ss:.1f} L {sx:.1f} {sy + ss:.1f} "
        f"M {sx - ss:.1f} {sy:.1f} L {sx + ss:.1f} {sy:.1f}"
    )
    elements.append(path(sparkle_d, fill="none", stroke=WHITE,
                         stroke_width=f"{SIZE * 0.012:.1f}",
                         stroke_linecap="round", opacity="0.85"))

    return svg_doc(SIZE, SIZE, *elements)


if __name__ == "__main__":
    svg = make_logo()
    out = "cpal-logo.svg"
    with open(out, "w") as f:
        f.write(svg)
    print(f"Wrote {out} ({len(svg)} bytes)")
