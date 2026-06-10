#!/usr/bin/env python3
"""Generate the cpal logo as SVG.

Concept: A warm coral circle containing a white magnifying glass
whose handle curves into a protective "c" shape. The lens contains
a </> code symbol — a helper that examines code carefully and keeps
you out of trouble.

v2: Better centered, curved handle, lens shine, refined brackets.
v3: Handle curves inward (c-shape), wider </> spacing, translucent lens.
"""

import math

# Anthropic-adjacent warm palette
CORAL = "#CC6B47"       # slightly richer terracotta
CORAL_DARK = "#A8523A"  # shadow tone
WHITE = "#FFFFFF"

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


def line(x1, y1, x2, y2, **attrs):
    a = " ".join(f'{k.replace("_", "-")}="{v}"' for k, v in attrs.items())
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" {a}/>'


def svg_doc(width, height, *children):
    inner = "\n  ".join(children)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">
  {inner}
</svg>"""


def make_logo():
    elements = []
    sw = SIZE * 0.042  # standard stroke width for glass ring + handle

    # Background circle — warm coral
    elements.append(circle(CX, CY, R, fill=CORAL))

    # === Magnifying glass ===
    # Lens centered slightly above and left of center so the handle
    # balances toward lower-right — feels natural in the circle.
    lens_cx = CX - SIZE * 0.02
    lens_cy = CY - SIZE * 0.05
    lens_r = SIZE * 0.17  # slightly larger lens

    # Handle — sweeps from ~5 o'clock on the lens in a gentle arc
    angle_start = math.radians(42)
    hx1 = lens_cx + (lens_r + sw * 0.3) * math.cos(angle_start)
    hy1 = lens_cy + (lens_r + sw * 0.3) * math.sin(angle_start)

    # End point — lower right, sweeping out to finish the "c"
    hx2 = CX + SIZE * 0.19
    hy2 = CY + SIZE * 0.24

    # Bezier controls — first control pulls straight down (vertical),
    # second pulls the end to sweep rightward — clean "c" arc
    cp1x = hx1 - SIZE * 0.02
    cp1y = hy1 + SIZE * 0.14
    cp2x = hx2 - SIZE * 0.10
    cp2y = hy2 + SIZE * 0.02

    handle_d = (
        f"M {hx1:.1f} {hy1:.1f} "
        f"C {cp1x:.1f} {cp1y:.1f}, {cp2x:.1f} {cp2y:.1f}, {hx2:.1f} {hy2:.1f}"
    )
    # Draw handle first (behind the ring)
    elements.append(path(handle_d, fill="none", stroke=WHITE,
                         stroke_width=f"{sw:.1f}",
                         stroke_linecap="round"))

    # Lens ring (thick white stroke, on top of handle)
    # fill="none" lets the background show through — translucent glass
    elements.append(circle(lens_cx, lens_cy, lens_r,
                           fill="none", stroke=WHITE,
                           stroke_width=f"{sw:.1f}"))

    # Lens glass tint — subtle lighter fill suggests glass
    elements.append(circle(lens_cx, lens_cy, lens_r * 0.85,
                           fill=WHITE, opacity="0.1"))

    # Specular highlight — small bright arc at upper-left of lens
    shine_r = lens_r * 0.65
    shine_cx = lens_cx - lens_r * 0.22
    shine_cy = lens_cy - lens_r * 0.22
    # Small crescent via two overlapping arcs
    sa = math.radians(-140)
    sb = math.radians(-60)
    ax1 = shine_cx + shine_r * math.cos(sa)
    ay1 = shine_cy + shine_r * math.sin(sa)
    ax2 = shine_cx + shine_r * math.cos(sb)
    ay2 = shine_cy + shine_r * math.sin(sb)
    shine_d = (
        f"M {ax1:.1f} {ay1:.1f} "
        f"A {shine_r:.1f} {shine_r:.1f} 0 0 1 {ax2:.1f} {ay2:.1f}"
    )
    elements.append(path(shine_d, fill="none", stroke=WHITE,
                         stroke_width=f"{SIZE * 0.012:.1f}",
                         stroke_linecap="round", opacity="0.35"))

    # === Code symbol inside the lens: < / > ===
    bsw = SIZE * 0.02  # bracket stroke width
    bracket_h = lens_r * 0.36  # half-height of brackets
    bracket_w = bracket_h * 0.5  # horizontal extent

    # Left bracket < — spread wider for breathing room
    lbx = lens_cx - lens_r * 0.36
    lb_d = (
        f"M {lbx:.1f} {lens_cy - bracket_h:.1f} "
        f"L {lbx - bracket_w:.1f} {lens_cy:.1f} "
        f"L {lbx:.1f} {lens_cy + bracket_h:.1f}"
    )
    elements.append(path(lb_d, fill="none", stroke=WHITE,
                         stroke_width=f"{bsw:.1f}",
                         stroke_linecap="round", stroke_linejoin="round"))

    # Slash /  (between the brackets, with room to breathe)
    slash_hw = lens_r * 0.07  # horizontal half-width of slash
    slash_hh = bracket_h * 0.7  # vertical half-height
    elements.append(line(
        f"{lens_cx + slash_hw:.1f}", f"{lens_cy - slash_hh:.1f}",
        f"{lens_cx - slash_hw:.1f}", f"{lens_cy + slash_hh:.1f}",
        stroke=WHITE, stroke_width=f"{bsw:.1f}", stroke_linecap="round"))

    # Right bracket >
    rbx = lens_cx + lens_r * 0.36
    rb_d = (
        f"M {rbx:.1f} {lens_cy - bracket_h:.1f} "
        f"L {rbx + bracket_w:.1f} {lens_cy:.1f} "
        f"L {rbx:.1f} {lens_cy + bracket_h:.1f}"
    )
    elements.append(path(rb_d, fill="none", stroke=WHITE,
                         stroke_width=f"{bsw:.1f}",
                         stroke_linecap="round", stroke_linejoin="round"))

    return svg_doc(SIZE, SIZE, *elements)


if __name__ == "__main__":
    svg = make_logo()
    out = "contrib/cpal-logo.svg"
    with open(out, "w") as f:
        f.write(svg)
    print(f"Wrote {out} ({len(svg)} bytes)")
