#!/usr/bin/env python3
"""
cpal "Building Together" — Git Timeline Video

Stop-motion animation telling the story of building cpal: 14 days, 26 commits,
from 0 to 1,342 LOC. Visualizes the git timeline and human-AI collaboration arc.

Created on Valentine's Day 2026 by Amy Tobey and Claude (Opus 4.6).

Rendering pipeline:
  1. Python generates 360 SVG frames (12fps x 30s) programmatically
  2. rsvg-convert rasterizes each SVG -> PNG (parallelized across 32 cores)
  3. ffmpeg assembles the PNG sequence + optional WAV audio -> H.264 MP4

The stop-motion aesthetic comes from per-frame random jitter (+-1-2px position
offset) and wobble (+-1deg rotation) on all elements, with deterministic seeds
so the animation is reproducible. Bounce easing with overshoot gives elements
a playful entrance.

Run:
    python contrib/git-timeline-video.py                    # full render with audio
    python contrib/git-timeline-video.py --no-audio         # visuals only
    python contrib/git-timeline-video.py --frame 150        # preview single frame

Requires: rsvg-convert (librsvg), ffmpeg, Python 3.10+
Produces: cpal-timeline.mp4 (1080x1080, ~30s)
"""

import argparse
import math
import os
import random
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# === Constants ===
WIDTH, HEIGHT = 1080, 1080
FPS = 12
DURATION = 53.0
TOTAL_FRAMES = int(FPS * DURATION)  # 636

FRAMES_DIR = Path("frames-timeline")
AUDIO_PATH = Path("cpal-narration.wav")
OUTPUT_PATH = Path("cpal-timeline.mp4")

# Seed for reproducible jitter (but different per frame)
random.seed(42)
JITTER_SEEDS = [random.randint(0, 999999) for _ in range(TOTAL_FRAMES + 1)]

# === Color Palette ===
COLORS = {
    # Scene 1: Day One — warm peach/amber
    "bg_top_1": "#FFF3E0",     # peach cream
    "bg_bot_1": "#FFE0B2",     # warm peach
    # Scene 2: Hardening — steel blue
    "bg_top_2": "#E3F2FD",     # ice blue
    "bg_bot_2": "#BBDEFB",     # steel blue
    # Scene 3: Growing Smarter — green/gold
    "bg_top_3": "#E8F5E9",     # soft green
    "bg_bot_3": "#FFF9C4",     # gold cream
    # Scene 4: Collaboration — lavender/coral
    "bg_top_4": "#F3E5F5",     # lavender
    "bg_bot_4": "#FCE4EC",     # coral blush
    # Scene 5: Kaizen — parchment darkening
    "bg_top_5": "#FFF8E1",     # parchment
    "bg_bot_5": "#F5F5DC",     # beige

    # Accent colors
    "amber": "#D97706",        # Claude amber
    "amber_light": "#F59E0B",
    "amber_dark": "#B45309",
    "gemini_blue": "#4285F4",  # Gemini blue
    "security_blue": "#1565C0",
    "shield_green": "#2E7D32",
    "bug_red": "#D32F2F",
    "bug_green": "#388E3C",
    "haiku_blue": "#42A5F5",
    "sonnet_violet": "#7E57C2",
    "opus_amber": "#D97706",
    "mit_green": "#4CAF50",

    # Common
    "card_bg": "#FFFFFF",
    "card_stroke": "#333333",
    "text_dark": "#2D2D2D",
    "text_light": "#FFFFFF",
    "text_muted": "#888888",
    "code_bg": "#1E1E2E",      # dark code card
    "code_text": "#A6E3A1",    # green code text
    "code_keyword": "#CBA6F7", # purple keywords
    "code_string": "#F9E2AF",  # yellow strings
    "sparkle": "#FFD700",
    "heart": "#E91E63",
}


# === Easing Functions ===

def ease_bounce(t: float) -> float:
    """Overshoot bounce easing: goes past 1.0 then settles."""
    if t < 0:
        return 0.0
    if t > 1:
        return 1.0
    return 1.0 - math.cos(t * math.pi * 0.5) * (1.0 - t) + 0.15 * math.sin(t * math.pi * 2) * (1.0 - t)


def ease_in_out(t: float) -> float:
    """Smooth ease in-out."""
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def ease_out_back(t: float) -> float:
    """Ease out with slight overshoot."""
    t = max(0.0, min(1.0, t))
    c1 = 1.70158
    c3 = c1 + 1
    return 1 + c3 * pow(t - 1, 3) + c1 * pow(t - 1, 2)


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * max(0.0, min(1.0, t))


def lerp_color(c1: str, c2: str, t: float) -> str:
    """Lerp between two hex colors."""
    t = max(0.0, min(1.0, t))
    r1, g1, b1 = int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)
    r2, g2, b2 = int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# === Jitter & Wobble ===

def jitter(frame: int, element_id: int = 0, amount: float = 1.5) -> tuple[float, float]:
    """Deterministic per-frame jitter."""
    rng = random.Random(JITTER_SEEDS[frame] + element_id * 7)
    return rng.uniform(-amount, amount), rng.uniform(-amount, amount)


def wobble(frame: int, element_id: int = 0, max_deg: float = 1.0) -> float:
    """Slight rotation wobble."""
    rng = random.Random(JITTER_SEEDS[frame] + element_id * 13)
    return rng.uniform(-max_deg, max_deg)


# === SVG Helpers ===

def svg_header(bg_top: str, bg_bot: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="{bg_top}"/>
      <stop offset="100%" stop-color="{bg_bot}"/>
    </linearGradient>
    <filter id="shadow" x="-5%" y="-5%" width="115%" height="115%">
      <feDropShadow dx="2" dy="3" stdDeviation="4" flood-opacity="0.15"/>
    </filter>
    <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="{WIDTH}" height="{HEIGHT}" fill="url(#bg)"/>'''


def svg_footer() -> str:
    return "</svg>"


def svg_rounded_rect(x: float, y: float, w: float, h: float,
                     rx: float = 20, fill: str = "#FFFFFF",
                     stroke: str = "#333333", stroke_width: float = 2,
                     opacity: float = 1.0, rotate: float = 0,
                     shadow: bool = True) -> str:
    transform = ""
    if rotate != 0:
        cx, cy = x + w / 2, y + h / 2
        transform = f' transform="rotate({rotate:.2f},{cx:.1f},{cy:.1f})"'
    filt = ' filter="url(#shadow)"' if shadow else ''
    return (f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'rx="{rx:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" '
            f'opacity="{opacity:.3f}"{filt}{transform}/>')


def svg_text(x: float, y: float, text: str, size: float = 48,
             fill: str = "#2D2D2D", anchor: str = "middle",
             weight: str = "bold", opacity: float = 1.0,
             rotate: float = 0, font: str = "sans-serif") -> str:
    transform = ""
    if rotate != 0:
        transform = f' transform="rotate({rotate:.2f},{x:.1f},{y:.1f})"'
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    return (f'  <text x="{x:.1f}" y="{y:.1f}" font-family="{font}" '
            f'font-size="{size:.1f}" font-weight="{weight}" fill="{fill}" '
            f'text-anchor="{anchor}" opacity="{opacity:.3f}"{transform}>{text}</text>')


def svg_sparkle(cx: float, cy: float, size: float = 12,
                fill: str = "#FFD700", opacity: float = 1.0) -> str:
    """Four-pointed sparkle star."""
    s = size
    points = [
        f"{cx},{cy - s}",
        f"{cx + s * 0.3},{cy - s * 0.3}",
        f"{cx + s},{cy}",
        f"{cx + s * 0.3},{cy + s * 0.3}",
        f"{cx},{cy + s}",
        f"{cx - s * 0.3},{cy + s * 0.3}",
        f"{cx - s},{cy}",
        f"{cx - s * 0.3},{cy - s * 0.3}",
    ]
    return (f'  <polygon points="{" ".join(points)}" '
            f'fill="{fill}" opacity="{opacity:.3f}"/>')


# === New SVG Helpers ===

def svg_code_bar(x: float, y: float, w: float, h: float, label: str,
                 fill: str = "#D97706", opacity: float = 1.0) -> str:
    """Rounded rect representing a file's LOC with filename label."""
    parts = []
    parts.append(f'  <rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
                 f'rx="6" fill="{fill}" opacity="{opacity:.3f}"/>')
    parts.append(svg_text(x + w / 2, y - 10, label, size=16, fill=COLORS["text_dark"],
                          weight="normal", opacity=opacity, font="monospace"))
    return "\n".join(parts)


def svg_code_snippet(x: float, y: float, w: float, lines: list[tuple[str, str]],
                     opacity: float = 1.0) -> str:
    """Dark code card with syntax-highlighted lines. lines = [(color, text), ...]"""
    h = 30 + len(lines) * 28
    parts = []
    parts.append(svg_rounded_rect(x, y, w, h, rx=12, fill=COLORS["code_bg"],
                                  stroke="#333344", stroke_width=1, opacity=opacity))
    # Window dots
    for i, color in enumerate(["#FF5F56", "#FFBD2E", "#27C93F"]):
        parts.append(f'  <circle cx="{x + 20 + i * 18:.1f}" cy="{y + 16:.1f}" r="5" '
                     f'fill="{color}" opacity="{opacity:.3f}"/>')
    # Code lines
    for i, (color, text) in enumerate(lines):
        text_escaped = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        ly = y + 48 + i * 28
        parts.append(f'  <text x="{x + 20:.1f}" y="{ly:.1f}" font-family="monospace" '
                     f'font-size="17" fill="{color}" text-anchor="start" '
                     f'opacity="{opacity:.3f}">{text_escaped}</text>')
    return "\n".join(parts)


def svg_shield(cx: float, cy: float, size: float = 50,
               fill: str = "#1565C0", opacity: float = 1.0) -> str:
    """Shield security icon."""
    s = size / 50
    return f'''  <g transform="translate({cx:.1f},{cy:.1f}) scale({s:.2f})" opacity="{opacity:.3f}">
    <path d="M0,-30 L25,-18 L25,8 Q25,28 0,35 Q-25,28 -25,8 L-25,-18 Z"
          fill="{fill}" stroke="#0D47A1" stroke-width="2"/>
    <path d="M-8,2 L-2,10 L10,-6" fill="none" stroke="white" stroke-width="3" stroke-linecap="round"/>
  </g>'''


def svg_lock(cx: float, cy: float, size: float = 50,
             fill: str = "#F57F17", opacity: float = 1.0) -> str:
    """Lock security icon."""
    s = size / 50
    return f'''  <g transform="translate({cx:.1f},{cy:.1f}) scale({s:.2f})" opacity="{opacity:.3f}">
    <rect x="-18" y="-5" width="36" height="28" rx="4" fill="{fill}" stroke="#E65100" stroke-width="2"/>
    <path d="M-10,-5 L-10,-15 Q-10,-28 0,-28 Q10,-28 10,-15 L10,-5"
          fill="none" stroke="{fill}" stroke-width="4"/>
    <circle cx="0" cy="10" r="4" fill="white"/>
    <rect x="-1.5" y="10" width="3" height="8" rx="1" fill="white"/>
  </g>'''


def svg_gears(cx: float, cy: float, size: float = 50,
              frame: int = 0, opacity: float = 1.0) -> str:
    """Thread-safety gears icon (two interlocking)."""
    s = size / 50
    rot1 = (frame * 5) % 360
    rot2 = -(frame * 5 + 22.5) % 360
    return f'''  <g transform="translate({cx:.1f},{cy:.1f}) scale({s:.2f})" opacity="{opacity:.3f}">
    <g transform="rotate({rot1:.1f},-10,-5)">
      <circle cx="-10" cy="-5" r="14" fill="#78909C" stroke="#455A64" stroke-width="2"/>
      <circle cx="-10" cy="-5" r="5" fill="#455A64"/>
      {"".join(f'<rect x="-12" y="-20" width="4" height="10" rx="2" fill="#78909C" transform="rotate({a},-10,-5)"/>' for a in range(0, 360, 45))}
    </g>
    <g transform="rotate({rot2:.1f},12,8)">
      <circle cx="12" cy="8" r="10" fill="#90A4AE" stroke="#546E7A" stroke-width="1.5"/>
      <circle cx="12" cy="8" r="4" fill="#546E7A"/>
      {"".join(f'<rect x="10" y="-5" width="4" height="8" rx="2" fill="#90A4AE" transform="rotate({a},12,8)"/>' for a in range(0, 360, 60))}
    </g>
  </g>'''


def svg_bug_dot(cx: float, cy: float, color: str = "#D32F2F",
                size: float = 10, opacity: float = 1.0) -> str:
    """Status dot (red=bug, green=fixed)."""
    return (f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{size:.1f}" '
            f'fill="{color}" opacity="{opacity:.3f}" filter="url(#glow)"/>')


def svg_tier_card(x: float, y: float, name: str, desc: str,
                  color: str, size_label: str,
                  opacity: float = 1.0, rotate: float = 0) -> str:
    """Model tier card with label and description."""
    parts = []
    parts.append(svg_rounded_rect(x, y, 280, 90, rx=15, fill="#FFFFFF",
                                  stroke=color, stroke_width=3,
                                  opacity=opacity, rotate=rotate))
    parts.append(svg_text(x + 140, y + 38, name, size=30, fill=color,
                          opacity=opacity))
    parts.append(svg_text(x + 140, y + 65, desc, size=16, fill=COLORS["text_muted"],
                          weight="normal", opacity=opacity))
    # Size indicator
    parts.append(svg_text(x + 260, y + 22, size_label, size=14, fill=color,
                          weight="normal", opacity=opacity * 0.7, anchor="end"))
    return "\n".join(parts)


def svg_loop_arrow(cx: float, cy: float, radius: float = 40,
                   progress: float = 1.0, stroke: str = "#D97706",
                   opacity: float = 1.0) -> str:
    """Circular progress arrow with stroke-dasharray draw-on."""
    circumference = 2 * math.pi * radius
    dash_len = circumference * min(1.0, progress) * 0.85  # leave gap for arrow
    gap_len = circumference - dash_len
    # Arrow head at end
    angle = progress * 0.85 * 2 * math.pi - math.pi / 2
    ax = cx + radius * math.cos(angle)
    ay = cy + radius * math.sin(angle)
    arrow_angle = math.degrees(angle) + 90
    return f'''  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{radius:.1f}"
    fill="none" stroke="{stroke}" stroke-width="4"
    stroke-dasharray="{dash_len:.1f} {gap_len:.1f}"
    stroke-dashoffset="0" stroke-linecap="round"
    transform="rotate(-90,{cx:.1f},{cy:.1f})"
    opacity="{opacity:.3f}"/>
  <polygon points="{ax:.1f},{ay - 6:.1f} {ax + 10:.1f},{ay:.1f} {ax:.1f},{ay + 6:.1f}"
    fill="{stroke}" opacity="{opacity:.3f}"
    transform="rotate({arrow_angle:.1f},{ax:.1f},{ay:.1f})"/>'''


def svg_line_chart(x: float, y: float, w: float, h: float,
                   data_points: list[tuple[float, float]],
                   progress: float = 1.0,
                   stroke: str = "#D97706", opacity: float = 1.0) -> str:
    """Polyline chart that draws left-to-right."""
    if not data_points:
        return ""
    # Normalize data
    max_y = max(d[1] for d in data_points) or 1
    points = []
    for dx, dy in data_points:
        px = x + dx * w
        py = y + h - (dy / max_y) * h
        points.append((px, py))

    # Only draw up to progress
    n_visible = max(1, int(len(points) * progress))
    visible = points[:n_visible]
    pts_str = " ".join(f"{px:.1f},{py:.1f}" for px, py in visible)

    parts = []
    # Axes
    parts.append(f'  <line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y + h:.1f}" '
                 f'stroke="#CCCCCC" stroke-width="1" opacity="{opacity:.3f}"/>')
    parts.append(f'  <line x1="{x:.1f}" y1="{y + h:.1f}" x2="{x + w:.1f}" y2="{y + h:.1f}" '
                 f'stroke="#CCCCCC" stroke-width="1" opacity="{opacity:.3f}"/>')
    # Line
    parts.append(f'  <polyline points="{pts_str}" fill="none" stroke="{stroke}" '
                 f'stroke-width="3" stroke-linecap="round" stroke-linejoin="round" '
                 f'opacity="{opacity:.3f}"/>')
    # Dots at data points
    for px, py in visible:
        parts.append(f'  <circle cx="{px:.1f}" cy="{py:.1f}" r="4" '
                     f'fill="{stroke}" opacity="{opacity:.3f}"/>')
    return "\n".join(parts)


def svg_model_node(cx: float, cy: float, label: str, color: str,
                   size: float = 40, opacity: float = 1.0) -> str:
    """Circle with centered label for bridge diagram."""
    parts = []
    parts.append(f'  <circle cx="{cx:.1f}" cy="{cy:.1f}" r="{size:.1f}" '
                 f'fill="{color}" opacity="{opacity:.3f}" filter="url(#shadow)"/>')
    parts.append(svg_text(cx, cy + size * 0.15, label, size=size * 0.4,
                          fill="white", opacity=opacity))
    return "\n".join(parts)


def svg_stat_card(x: float, y: float, number: str, label: str,
                  color: str = "#D97706", opacity: float = 1.0,
                  rotate: float = 0) -> str:
    """Large number + small label stat card."""
    parts = []
    parts.append(svg_rounded_rect(x, y, 200, 110, rx=16, fill="#FFFFFF",
                                  stroke=color, stroke_width=2,
                                  opacity=opacity, rotate=rotate))
    parts.append(svg_text(x + 100, y + 55, number, size=42, fill=color,
                          opacity=opacity))
    parts.append(svg_text(x + 100, y + 85, label, size=16, fill=COLORS["text_muted"],
                          weight="normal", opacity=opacity))
    return "\n".join(parts)


def svg_arrow(x1: float, y1: float, x2: float, y2: float,
              stroke: str = "#666666", width: float = 2,
              opacity: float = 1.0, dashed: bool = False) -> str:
    """Arrow line with arrowhead."""
    angle = math.atan2(y2 - y1, x2 - x1)
    # Arrowhead points
    head_len = 10
    ax1 = x2 - head_len * math.cos(angle - 0.4)
    ay1 = y2 - head_len * math.sin(angle - 0.4)
    ax2 = x2 - head_len * math.cos(angle + 0.4)
    ay2 = y2 - head_len * math.sin(angle + 0.4)

    dash = ' stroke-dasharray="8,4"' if dashed else ''
    parts = []
    parts.append(f'  <line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{stroke}" stroke-width="{width}"{dash} opacity="{opacity:.3f}"/>')
    parts.append(f'  <polygon points="{x2:.1f},{y2:.1f} {ax1:.1f},{ay1:.1f} {ax2:.1f},{ay2:.1f}" '
                 f'fill="{stroke}" opacity="{opacity:.3f}"/>')
    return "\n".join(parts)


# === Scene Renderers ===

def scene_day_one(frame: int, local_frame: int, total_scene_frames: int) -> str:
    """Scene 1: 'Day One' — title, first commit visualization, file tree (0-179, 15s)."""
    parts = [svg_header(COLORS["bg_top_1"], COLORS["bg_bot_1"])]

    # --- Phase 1: Title (0-3s, frames 0-35) ---
    enter_t = ease_out_back(min(1.0, local_frame / 30))
    title_y = lerp(-100, 200, enter_t)
    jx, jy = jitter(frame, 0)
    w = wobble(frame, 0, 0.8)
    parts.append(svg_text(540 + jx, title_y + jy, "cpal", size=140,
                          fill=COLORS["amber"], rotate=w))

    # Date slides in
    date_enter = ease_out_back(min(1.0, max(0, local_frame - 12) / 24))
    date_x = lerp(1200, 540, date_enter)
    jx2, jy2 = jitter(frame, 1)
    parts.append(svg_text(date_x + jx2, 280 + jy2, "January 31, 2026", size=36,
                          fill=COLORS["text_muted"], weight="normal"))

    # --- Phase 2: Code bars (3-6s, frames 36-71) ---
    if local_frame > 30:
        bar_t = ease_in_out(min(1.0, (local_frame - 30) / 28))
        files = [
            ("server.py", 559, COLORS["amber"]),
            ("__init__.py", 8, COLORS["amber_light"]),
            ("test_tools.py", 320, "#66BB6A"),
            ("test_connectivity.py", 45, "#81C784"),
            ("test_agentic.py", 85, "#A5D6A7"),
            ("pyproject.toml", 62, "#90A4AE"),
        ]
        max_loc = 559
        bar_x_start = 100
        bar_width = 90
        bar_gap = 15
        bar_bottom = 620

        for i, (fname, loc, color) in enumerate(files):
            bx = bar_x_start + i * (bar_width + bar_gap)
            delay = i * 0.08
            delayed_t = ease_out_back(min(1.0, max(0, bar_t - delay) / (1 - delay + 0.01)))
            actual_h = (loc / max_loc) * 250 * delayed_t
            actual_y = bar_bottom - actual_h
            bjx, bjy = jitter(frame, 10 + i)
            parts.append(svg_code_bar(bx + bjx, actual_y + bjy,
                                      bar_width, actual_h, fname, color,
                                      opacity=ease_in_out(min(1.0, (local_frame - 30) / 14))))

        # LOC counter
        if local_frame > 40:
            loc_t = ease_in_out(min(1.0, (local_frame - 40) / 20))
            loc_count = int(1189 * loc_t)
            ljx, ljy = jitter(frame, 20)
            parts.append(svg_text(800 + ljx, 500 + ljy, f"{loc_count:,}", size=56,
                                  fill=COLORS["amber"]))
            parts.append(svg_text(800 + ljx, 540 + ljy, "lines of code", size=20,
                                  fill=COLORS["text_muted"], weight="normal"))

    # --- Phase 3: File tree (6-9s, frames 72-107) ---
    if local_frame > 65:
        tree_t = ease_in_out(min(1.0, (local_frame - 65) / 28))
        tree_lines = [
            "cpal/",
            "  src/cpal/",
            "    __init__.py",
            "    server.py",
            "  tests/",
            "    test_tools.py",
            "  pyproject.toml",
        ]
        tree_x = 120
        tree_y = 680
        n_visible = max(1, int(len(tree_lines) * tree_t))
        for i in range(n_visible):
            line_opacity = ease_in_out(min(1.0, (tree_t * len(tree_lines) - i)))
            tjx, tjy = jitter(frame, 30 + i)
            parts.append(svg_text(tree_x + tjx, tree_y + i * 28 + tjy,
                                  tree_lines[i], size=20,
                                  fill=COLORS["text_dark"], anchor="start",
                                  font="monospace", weight="normal",
                                  opacity=line_opacity))

    # --- Phase 4: Code snippet + tagline (10-13s, frames 120-155) ---
    if local_frame > 100:
        code_t = ease_out_back(min(1.0, (local_frame - 100) / 24))
        code_y = lerp(1100, 680, code_t)
        code_opacity = ease_in_out(min(1.0, (local_frame - 100) / 18))
        parts.append(svg_code_snippet(380, code_y, 620, [
            (COLORS["code_keyword"], "def consult_claude(query, model=\"opus\"):"),
            (COLORS["code_text"],    "    \"\"\"Your pal Claude, ready to help.\"\"\""),
        ], opacity=code_opacity))

    if local_frame > 120:
        tag_opacity = ease_in_out(min(1.0, (local_frame - 120) / 16))
        tjx, tjy = jitter(frame, 40)
        parts.append(svg_text(540 + tjx, 960 + tjy, "Production-ready. Day one.",
                              size=36, fill=COLORS["amber_dark"],
                              opacity=tag_opacity))

    # Sparkle burst at end
    if local_frame > 140:
        sparkle_opacity = ease_in_out(min(1.0, (local_frame - 140) / 14))
        sparkle_positions = [(180, 150), (900, 130), (120, 380), (950, 400),
                             (200, 900), (880, 920)]
        for i, (sx, sy) in enumerate(sparkle_positions):
            phase = math.sin(frame * 0.4 + i * 1.2)
            s_size = 8 + 6 * (0.5 + 0.5 * phase)
            sjx, sjy = jitter(frame, 50 + i, 2)
            parts.append(svg_sparkle(sx + sjx, sy + sjy, s_size,
                                     fill=COLORS["sparkle"],
                                     opacity=sparkle_opacity * (0.5 + 0.5 * phase)))

    parts.append(svg_footer())
    return "\n".join(parts)


def scene_hardening(frame: int, local_frame: int, total_scene_frames: int) -> str:
    """Scene 2: 'Hardening' — security cards, bug dots, MIT badge (180-287, 9s)."""
    parts = [svg_header(COLORS["bg_top_2"], COLORS["bg_bot_2"])]

    # --- Title ---
    jx, jy = jitter(frame, 0)
    enter_t = ease_out_back(min(1.0, local_frame / 22))
    title_y = lerp(-60, 100, enter_t)
    parts.append(svg_text(540 + jx, title_y + jy, "Hardening", size=64,
                          fill=COLORS["security_blue"]))
    if local_frame > 10:
        sub_opacity = ease_in_out(min(1.0, (local_frame - 10) / 14))
        parts.append(svg_text(540 + jx, title_y + 45 + jy,
                              "14 commits on day one",
                              size=24, fill=COLORS["text_muted"],
                              weight="normal", opacity=sub_opacity))

    # Timeline bar
    if local_frame > 12:
        bar_t = ease_in_out(min(1.0, (local_frame - 12) / 18))
        bar_w = 700 * bar_t
        parts.append(f'  <rect x="{190:.1f}" y="170" width="{bar_w:.1f}" height="6" '
                     f'rx="3" fill="{COLORS["security_blue"]}" opacity="0.4"/>')
        for i in range(int(14 * bar_t)):
            tx = 190 + i * 50
            parts.append(f'  <rect x="{tx:.1f}" y="164" width="3" height="18" '
                         f'rx="1" fill="{COLORS["security_blue"]}" opacity="0.7"/>')

    # --- Security cards ---
    if local_frame > 16:
        card_defs = [
            ("shield", "Path Traversal\nProtection", COLORS["security_blue"]),
            ("lock", "Symlink Attack\nPrevention", "#F57F17"),
            ("gears", "Thread Safety", "#546E7A"),
        ]
        for i, (icon_type, label, color) in enumerate(card_defs):
            card_enter = ease_out_back(min(1.0, max(0, local_frame - 16 - i * 8) / 20))
            cx = 180 + i * 270
            card_y = lerp(1100, 240, card_enter)
            cjx, cjy = jitter(frame, 10 + i)
            cw = wobble(frame, 10 + i, 0.6)
            card_opacity = ease_in_out(min(1.0, max(0, local_frame - 16 - i * 8) / 12))

            parts.append(svg_rounded_rect(cx + cjx - 110, card_y + cjy, 220, 180,
                                          rx=16, fill="#FFFFFF",
                                          stroke=color, stroke_width=2,
                                          opacity=card_opacity, rotate=cw))
            icon_cx = cx + cjx
            icon_cy = card_y + cjy + 60
            if icon_type == "shield":
                parts.append(svg_shield(icon_cx, icon_cy, 45, opacity=card_opacity))
            elif icon_type == "lock":
                parts.append(svg_lock(icon_cx, icon_cy, 45, opacity=card_opacity))
            else:
                parts.append(svg_gears(icon_cx, icon_cy, 45, frame, card_opacity))
            for j, line in enumerate(label.split("\n")):
                parts.append(svg_text(cx + cjx, card_y + cjy + 120 + j * 22, line,
                                      size=17, fill=COLORS["text_dark"],
                                      weight="normal", opacity=card_opacity))

    # --- Code snippet ---
    if local_frame > 35:
        code_opacity = ease_in_out(min(1.0, (local_frame - 35) / 14))
        parts.append(svg_code_snippet(120, 470, 580, [
            (COLORS["code_keyword"], "if not resolved.is_relative_to("),
            (COLORS["code_string"],  "        project_root):"),
            (COLORS["code_text"],    "    raise ValueError(\"blocked\")"),
        ], opacity=code_opacity))

    # --- Bug dots ---
    if local_frame > 46:
        bug_positions = [(760, 500), (810, 540), (780, 580), (830, 510),
                         (850, 560), (770, 550), (820, 590)]
        for i, (bx, by) in enumerate(bug_positions):
            bug_enter = ease_out_back(min(1.0, max(0, local_frame - 46 - i * 2) / 12))
            bjx, bjy = jitter(frame, 40 + i)
            fix_t = max(0, min(1.0, (local_frame - 65 - i * 1.5) / 10))
            color = lerp_color(COLORS["bug_red"], COLORS["bug_green"], fix_t)
            parts.append(svg_bug_dot(bx + bjx, by + bjy, color,
                                     size=8 * bug_enter, opacity=bug_enter))

        if local_frame > 52:
            label_opacity = ease_in_out(min(1.0, (local_frame - 52) / 10))
            parts.append(svg_text(805, 480, "7 bugs found", size=18,
                                  fill=COLORS["text_muted"], weight="normal",
                                  opacity=label_opacity))
        if local_frame > 74:
            fixed_opacity = ease_in_out(min(1.0, (local_frame - 74) / 10))
            parts.append(svg_text(805, 625, "all fixed", size=18,
                                  fill=COLORS["shield_green"], weight="normal",
                                  opacity=fixed_opacity))

    # --- MIT badge ---
    if local_frame > 84:
        mit_t = ease_out_back(min(1.0, (local_frame - 84) / 14))
        mjx, mjy = jitter(frame, 50)
        badge_y = lerp(1100, 720, mit_t)
        parts.append(svg_rounded_rect(420 + mjx, badge_y + mjy, 240, 60,
                                      rx=30, fill=COLORS["mit_green"],
                                      stroke="#2E7D32", stroke_width=2,
                                      opacity=mit_t))
        parts.append(svg_text(540 + mjx, badge_y + 40 + mjy, "MIT Licensed",
                              size=26, fill="white", opacity=mit_t))

        if local_frame > 92:
            for i in range(4):
                phase = math.sin(frame * 0.5 + i * 1.5)
                if phase > 0.3:
                    sx = 400 + i * 80
                    sy = badge_y + 30 + 30 * math.sin(frame * 0.3 + i)
                    sjx, sjy = jitter(frame, 60 + i)
                    parts.append(svg_sparkle(sx + sjx, sy + sjy, 8,
                                             fill=COLORS["sparkle"],
                                             opacity=0.6 * phase))

    parts.append(svg_footer())
    return "\n".join(parts)


def scene_growing(frame: int, local_frame: int, total_scene_frames: int) -> str:
    """Scene 3: 'Growing Smarter' — model tiers, self-review, LOC chart (288-443, 13s)."""
    parts = [svg_header(COLORS["bg_top_3"], COLORS["bg_bot_3"])]

    # --- Model tier cards (frames 0-35) ---
    tiers = [
        ("haiku", "Fast exploration", COLORS["haiku_blue"], "quick"),
        ("sonnet", "Balanced reasoning", COLORS["sonnet_violet"], "mid"),
        ("opus", "Deep analysis", COLORS["opus_amber"], "deep"),
    ]
    for i, (name, desc, color, size_label) in enumerate(tiers):
        card_enter = ease_out_back(min(1.0, max(0, local_frame - i * 8) / 24))
        cx = 80 + i * 310
        card_y = lerp(-120, 60, card_enter)
        cjx, cjy = jitter(frame, i)
        cw = wobble(frame, i, 0.5)
        parts.append(svg_tier_card(cx + cjx, card_y + cjy, name, desc, color,
                                   size_label, opacity=card_enter, rotate=cw))

    # --- Extended thinking code snippet ---
    if local_frame > 24:
        code_opacity = ease_in_out(min(1.0, (local_frame - 24) / 14))
        parts.append(svg_code_snippet(60, 190, 500, [
            (COLORS["code_keyword"], '"thinking": {'),
            (COLORS["code_string"],  '    "type": "enabled",'),
            (COLORS["code_text"],    '    "budget_tokens": 50000'),
            (COLORS["code_keyword"], '}'),
        ], opacity=code_opacity))

    # --- Self-review loop (frames 34-65) ---
    if local_frame > 34:
        loop_t = ease_in_out(min(1.0, (local_frame - 34) / 24))
        loop_cx, loop_cy = 780, 310
        ljx, ljy = jitter(frame, 10)

        arrow_progress = min(1.0, (local_frame - 34) / 36)
        parts.append(svg_loop_arrow(loop_cx + ljx, loop_cy + ljy, 55,
                                    arrow_progress, COLORS["amber"],
                                    opacity=loop_t))

        parts.append(svg_text(loop_cx + ljx, loop_cy + ljy + 8, "{...}",
                              size=22, fill=COLORS["text_dark"], font="monospace",
                              opacity=loop_t))

        passes = [
            ("7 bugs", COLORS["bug_red"]),
            ("v0.2.0", COLORS["amber"]),
            ("6 more", COLORS["bug_red"]),
        ]
        for i, (label, color) in enumerate(passes):
            pass_t = max(0, min(1.0, (local_frame - 42 - i * 10) / 10))
            pjx, pjy = jitter(frame, 20 + i)
            py = 250 + i * 50
            parts.append(svg_text(930 + pjx, py + pjy, label, size=18,
                                  fill=color, weight="normal",
                                  opacity=pass_t, anchor="start"))
            parts.append(svg_text(910 + pjx, py + pjy, f"#{i+1}", size=14,
                                  fill=COLORS["text_muted"], weight="normal",
                                  opacity=pass_t, anchor="end"))

    # --- LOC growth chart (frames 56-143) ---
    if local_frame > 56:
        chart_t = ease_in_out(min(1.0, (local_frame - 56) / 36))
        chart_x, chart_y = 100, 470
        chart_w, chart_h = 860, 280

        parts.append(svg_rounded_rect(chart_x - 20, chart_y - 40, chart_w + 60, chart_h + 80,
                                      rx=16, fill="#FFFFFF",
                                      stroke="#E0E0E0", stroke_width=1,
                                      opacity=chart_t * 0.9, shadow=False))
        parts.append(svg_text(chart_x + chart_w / 2, chart_y - 10,
                              "Lines of Code", size=20, fill=COLORS["text_muted"],
                              weight="normal", opacity=chart_t))

        data = [
            (0.0, 1189),   # Jan 31
            (0.07, 1189),
            (0.14, 1189),
            (0.28, 1210),  # Feb 3
            (0.42, 1240),  # Feb 6
            (0.57, 1280),  # Feb 9
            (0.71, 1310),  # Feb 11
            (0.85, 1330),  # Feb 12
            (1.0, 1342),   # Feb 13
        ]
        chart_progress = min(1.0, (local_frame - 56) / 55)
        parts.append(svg_line_chart(chart_x, chart_y, chart_w, chart_h,
                                    data, progress=chart_progress,
                                    stroke=COLORS["amber"], opacity=chart_t))

        if chart_t > 0.5:
            label_opacity = (chart_t - 0.5) * 2
            parts.append(svg_text(chart_x, chart_y + chart_h + 25, "Jan 31",
                                  size=16, fill=COLORS["text_muted"], anchor="start",
                                  weight="normal", opacity=label_opacity))
            parts.append(svg_text(chart_x + chart_w, chart_y + chart_h + 25, "Feb 13",
                                  size=16, fill=COLORS["text_muted"], anchor="end",
                                  weight="normal", opacity=label_opacity))

        if chart_progress > 0.3:
            parts.append(svg_text(chart_x + chart_w * 0.5, chart_y + chart_h + 50,
                                  "v0.1.0 -> v0.2.0 -> v0.2.2", size=18,
                                  fill=COLORS["amber"], weight="normal",
                                  opacity=min(1.0, (chart_progress - 0.3) / 0.3)))

    parts.append(svg_footer())
    return "\n".join(parts)


def scene_collaboration(frame: int, local_frame: int, total_scene_frames: int) -> str:
    """Scene 4: 'The Collaboration' — bridge diagram, stats, cross-review (444-563, 10s)."""
    parts = [svg_header(COLORS["bg_top_4"], COLORS["bg_bot_4"])]

    # --- Bridge diagram (frames 0-35) ---
    bridge_t = ease_out_back(min(1.0, local_frame / 30))

    # gpal node (left)
    gpal_x = lerp(-100, 220, bridge_t)
    gjx, gjy = jitter(frame, 0)
    parts.append(svg_model_node(gpal_x + gjx, 180 + gjy, "gpal",
                                COLORS["gemini_blue"], 55, opacity=bridge_t))
    parts.append(svg_text(gpal_x + gjx, 260 + gjy, "Gemini", size=18,
                          fill=COLORS["text_muted"], weight="normal",
                          opacity=bridge_t))

    # MCP hub (center)
    mcp_y = lerp(-100, 180, bridge_t)
    mjx, mjy = jitter(frame, 1)
    parts.append(svg_rounded_rect(480 + mjx, mcp_y + mjy - 30, 120, 60,
                                  rx=12, fill="#FFFFFF", stroke="#9E9E9E",
                                  opacity=bridge_t))
    parts.append(svg_text(540 + mjx, mcp_y + mjy + 8, "MCP", size=24,
                          fill=COLORS["text_dark"], opacity=bridge_t))

    # cpal node (right)
    cpal_x = lerp(1200, 860, bridge_t)
    cjx, cjy = jitter(frame, 2)
    parts.append(svg_model_node(cpal_x + cjx, 180 + cjy, "cpal",
                                COLORS["amber"], 55, opacity=bridge_t))
    parts.append(svg_text(cpal_x + cjx, 260 + cjy, "Claude", size=18,
                          fill=COLORS["text_muted"], weight="normal",
                          opacity=bridge_t))

    # Arrows
    if local_frame > 14:
        arrow_opacity = ease_in_out(min(1.0, (local_frame - 14) / 12))
        parts.append(svg_arrow(gpal_x + 60, 180, 480, 180,
                               stroke=COLORS["gemini_blue"], opacity=arrow_opacity))
        parts.append(svg_arrow(600, 180, cpal_x - 60, 180,
                               stroke=COLORS["amber"], opacity=arrow_opacity))
        parts.append(svg_arrow(cpal_x - 60, 195, 600, 195,
                               stroke=COLORS["amber"], opacity=arrow_opacity * 0.5,
                               dashed=True))
        parts.append(svg_arrow(480, 195, gpal_x + 60, 195,
                               stroke=COLORS["gemini_blue"], opacity=arrow_opacity * 0.5,
                               dashed=True))

    # --- Stats cascade (frames 34-75) ---
    if local_frame > 34:
        stats = [
            ("42", "projects", COLORS["gemini_blue"]),
            ("35", "sessions", COLORS["sonnet_violet"]),
            ("26", "commits", COLORS["amber"]),
            ("14", "days", COLORS["shield_green"]),
        ]
        for i, (number, label, color) in enumerate(stats):
            stat_enter = ease_out_back(min(1.0, max(0, local_frame - 34 - i * 5) / 18))
            sx = 50 + i * 255
            stat_y = lerp(1100, 340, stat_enter)
            sjx, sjy = jitter(frame, 10 + i)
            sw = wobble(frame, 10 + i, 0.5)
            parts.append(svg_stat_card(sx + sjx, stat_y + sjy, number, label,
                                       color, opacity=stat_enter, rotate=sw))

    # --- Cross-review flow (frames 64-119) ---
    if local_frame > 64:
        review_t = ease_in_out(min(1.0, (local_frame - 64) / 22))

        parts.append(svg_code_snippet(100, 510, 540, [
            (COLORS["code_keyword"], "consult_claude("),
            (COLORS["code_string"],  '    query="Review server.py..."'),
            (COLORS["code_text"],    '    extended_thinking=True'),
            (COLORS["code_keyword"], ")"),
        ], opacity=review_t))

        flow_labels = [
            ("Gemini Pro reviews cpal", COLORS["gemini_blue"]),
            ("Claude reviews feedback", COLORS["amber"]),
        ]
        for i, (label, color) in enumerate(flow_labels):
            flow_t = max(0, min(1.0, (local_frame - 70 - i * 10) / 12))
            fjx, fjy = jitter(frame, 30 + i)
            parts.append(svg_text(830 + fjx, 560 + i * 45 + fjy, label,
                                  size=20, fill=color, weight="normal",
                                  opacity=flow_t))

    if local_frame > 84:
        flow_arrow_opacity = ease_in_out(min(1.0, (local_frame - 84) / 10))
        parts.append(svg_arrow(760, 565, 760, 590, stroke="#CCCCCC",
                               opacity=flow_arrow_opacity))

    # Sparkles
    if local_frame > 92:
        for i in range(5):
            phase = math.sin(frame * 0.4 + i * 1.3)
            if phase > 0.2:
                sx = 100 + i * 220
                sy = 700 + 20 * math.sin(frame * 0.2 + i)
                sjx, sjy = jitter(frame, 40 + i)
                parts.append(svg_sparkle(sx + sjx, sy + sjy, 8 + 4 * phase,
                                         fill=COLORS["sparkle"],
                                         opacity=0.5 * phase))

    parts.append(svg_footer())
    return "\n".join(parts)


def scene_kaizen(frame: int, local_frame: int, total_scene_frames: int) -> str:
    """Scene 5: 'Kaizen' finale — kanji, versions, valentine heart (564-635, 6s)."""
    t = local_frame / total_scene_frames
    # Darken background as scene progresses
    bg_top = lerp_color(COLORS["bg_top_5"], "#2D1B00", t * 0.6)
    bg_bot = lerp_color(COLORS["bg_bot_5"], "#1a0f00", t * 0.6)
    parts = [svg_header(bg_top, bg_bot)]

    text_color = lerp_color(COLORS["text_dark"], "#FFFFFF", t * 0.8)
    muted_color = lerp_color(COLORS["text_muted"], "#CCCCCC", t * 0.8)

    # --- Kaizen kanji (frames 0-28) ---
    kaizen_enter = ease_out_back(min(1.0, local_frame / 24))
    jx, jy = jitter(frame, 0)
    w = wobble(frame, 0, 0.5)
    kaizen_y = lerp(-100, 320, kaizen_enter)
    parts.append(svg_text(540 + jx, kaizen_y + jy, "改善", size=160,
                          fill=COLORS["amber"], opacity=kaizen_enter, rotate=w))

    # Subtitle
    if local_frame > 12:
        sub_opacity = ease_in_out(min(1.0, (local_frame - 12) / 14))
        sjx, sjy = jitter(frame, 1)
        parts.append(svg_text(540 + sjx, kaizen_y + 80 + sjy,
                              "continuous improvement",
                              size=30, fill=muted_color, weight="normal",
                              opacity=sub_opacity))

    # --- Version ticker + credits (frames 28-60) ---
    if local_frame > 28:
        ver_t = ease_in_out(min(1.0, (local_frame - 28) / 18))
        versions = ["v0.1.0", "v0.2.0", "v0.2.2"]
        for i, ver in enumerate(versions):
            ver_opacity = max(0, min(1.0, (ver_t * len(versions) - i)))
            vjx, vjy = jitter(frame, 10 + i)
            vx = 280 + i * 200
            parts.append(svg_rounded_rect(vx + vjx - 70, 500 + vjy, 140, 45,
                                          rx=22, fill=COLORS["amber"],
                                          stroke=COLORS["amber_dark"],
                                          opacity=ver_opacity))
            parts.append(svg_text(vx + vjx, 530 + vjy, ver, size=22,
                                  fill="white", opacity=ver_opacity))

        if ver_t > 0.4:
            for i in range(2):
                arrow_opacity = max(0, min(1.0, (ver_t - 0.3 - i * 0.2) * 3))
                ax = 350 + i * 200
                parts.append(svg_text(ax, 525, "->", size=24, fill=text_color,
                                      opacity=arrow_opacity, weight="normal"))

    # Credits
    if local_frame > 38:
        credit_opacity = ease_in_out(min(1.0, (local_frame - 38) / 14))
        cjx, cjy = jitter(frame, 20)
        parts.append(svg_text(540 + cjx, 610 + cjy, "Built by Amy & Claude",
                              size=28, fill=text_color, weight="normal",
                              opacity=credit_opacity))

    # Valentine heart
    if local_frame > 44:
        heart_opacity = ease_in_out(min(1.0, (local_frame - 44) / 12))
        hjx, hjy = jitter(frame, 21)
        bob = math.sin(frame * 0.3) * 6
        pulse = 1.0 + 0.08 * math.sin(frame * 0.5)
        heart_y = 670 + bob + hjy
        parts.append(f'  <g transform="translate({540 + hjx:.1f},{heart_y:.1f}) scale({pulse:.2f})" '
                     f'opacity="{heart_opacity:.3f}">')
        parts.append(f'    <path d="M0,-15 C-20,-40 -50,-15 -30,10 C-20,25 0,40 0,40 '
                     f'C0,40 20,25 30,10 C50,-15 20,-40 0,-15Z" '
                     f'fill="{COLORS["heart"]}"/>')
        parts.append('  </g>')

    # --- Finale text (frames 52-71) ---
    if local_frame > 52:
        finale_opacity = ease_in_out(min(1.0, (local_frame - 52) / 12))
        fjx, fjy = jitter(frame, 30)
        parts.append(svg_text(540 + fjx, 790 + fjy,
                              "改善の旅は続きます",
                              size=40, fill=COLORS["amber"],
                              opacity=finale_opacity))
        if local_frame > 58:
            trans_opacity = ease_in_out(min(1.0, (local_frame - 58) / 10))
            parts.append(svg_text(540 + fjx, 840 + fjy,
                                  "The journey of improvement continues",
                                  size=20, fill=muted_color, weight="normal",
                                  opacity=trans_opacity))

    # Sparkle burst
    if local_frame > 40:
        burst_t = (local_frame - 40) / max(1, total_scene_frames - 40)
        num_sparkles = 20
        for i in range(num_sparkles):
            angle = (2 * math.pi * i / num_sparkles) + frame * 0.03
            radius = 60 + burst_t * 400
            sx = 540 + math.cos(angle) * radius
            sy = 500 + math.sin(angle) * radius
            phase = math.sin(frame * 0.4 + i * 0.6)
            s_size = 5 + 7 * max(0, phase)
            s_opacity = max(0, 0.7 - burst_t * 0.4) * max(0, phase)
            sjx, sjy = jitter(frame, 50 + i)
            parts.append(svg_sparkle(sx + sjx, sy + sjy, s_size,
                                     fill=lerp_color(COLORS["sparkle"], COLORS["amber"],
                                                     burst_t * 0.5),
                                     opacity=s_opacity))

    # Gentle fade at the very end
    if local_frame > 64:
        fade = ease_in_out(min(1.0, (local_frame - 64) / 7))
        parts.append(f'  <rect width="{WIDTH}" height="{HEIGHT}" '
                     f'fill="#000000" opacity="{fade * 0.5:.3f}"/>')

    parts.append(svg_footer())
    return "\n".join(parts)


# === Frame Generation ===

# Scene boundaries (frame ranges, inclusive) — 53s at 12fps = 636 frames
SCENES = [
    (0, 179, scene_day_one, 180),            # 0-15s
    (180, 287, scene_hardening, 108),        # 15-24s
    (288, 443, scene_growing, 156),          # 24-37s
    (444, 563, scene_collaboration, 120),    # 37-47s
    (564, TOTAL_FRAMES - 1, scene_kaizen, TOTAL_FRAMES - 564),  # 47-53s
]


def generate_svg(frame_num: int) -> str:
    """Generate SVG markup for a single frame."""
    for start, end, scene_fn, total in SCENES:
        if start <= frame_num <= end:
            return scene_fn(frame_num, frame_num - start, total)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}">
  <rect width="{WIDTH}" height="{HEIGHT}" fill="black"/>
</svg>'''


def render_single_frame(args: tuple[int, str]) -> int:
    """Render one frame: generate SVG, pipe to rsvg-convert, save PNG."""
    frame_num, output_dir = args
    svg = generate_svg(frame_num)
    png_path = os.path.join(output_dir, f"{frame_num:04d}.png")

    proc = subprocess.run(
        ["rsvg-convert", "--width", str(WIDTH), "--height", str(HEIGHT),
         "--format", "png", "--output", png_path],
        input=svg.encode("utf-8"),
        capture_output=True,
    )
    if proc.returncode != 0:
        print(f"  x Frame {frame_num}: {proc.stderr.decode()}", file=sys.stderr)
        return -1
    return frame_num


def render_frames(output_dir: str, total_frames: int) -> None:
    """Generate all frames in parallel."""
    os.makedirs(output_dir, exist_ok=True)

    tasks = [(i, output_dir) for i in range(total_frames)]
    workers = min(32, os.cpu_count() or 4)

    print(f"Rendering {total_frames} frames using {workers} workers...")
    completed = 0
    errors = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(render_single_frame, t): t[0] for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result == -1:
                errors += 1
            else:
                completed += 1
            if completed % 50 == 0:
                print(f"  {completed}/{total_frames} frames rendered")

    print(f"Rendered {completed} frames ({errors} errors)")
    if errors > 0:
        print("Some frames had errors", file=sys.stderr)


def assemble_video(frames_dir: str, output_path: str,
                   audio_path: str | None = None) -> None:
    """Use ffmpeg to combine frames + optional audio into mp4."""
    print("Assembling video...")
    cmd = [
        "ffmpeg",
        "-framerate", str(FPS),
        "-i", f"{frames_dir}/%04d.png",
    ]
    if audio_path and Path(audio_path).exists():
        cmd.extend(["-i", audio_path])

    cmd.extend([
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
    ])

    if audio_path and Path(audio_path).exists():
        cmd.extend(["-c:a", "aac", "-shortest"])

    cmd.extend(["-y", output_path])

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"ffmpeg failed:\n{proc.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"Video saved to {output_path}")


def preview_frame(frame_num: int) -> None:
    """Render and save a single frame as SVG + PNG for preview."""
    svg = generate_svg(frame_num)
    svg_path = f"preview-frame-{frame_num:04d}.svg"
    png_path = f"preview-frame-{frame_num:04d}.png"

    with open(svg_path, "w") as f:
        f.write(svg)
    print(f"SVG saved to {svg_path}")

    proc = subprocess.run(
        ["rsvg-convert", "--width", str(WIDTH), "--height", str(HEIGHT),
         "--format", "png", "--output", png_path],
        input=svg.encode("utf-8"),
        capture_output=True,
    )
    if proc.returncode == 0:
        print(f"PNG saved to {png_path}")
    else:
        print(f"rsvg-convert failed: {proc.stderr.decode()}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="cpal 'Building Together' git timeline video generator")
    parser.add_argument("--no-audio", action="store_true",
                        help="Render video without audio")
    parser.add_argument("--frame", type=int, default=None,
                        help="Preview a single frame number")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH),
                        help="Output video path")
    parser.add_argument("--audio", type=str, default=str(AUDIO_PATH),
                        help="Audio file path")
    args = parser.parse_args()

    if args.frame is not None:
        print(f"Previewing frame {args.frame} (scene: ", end="")
        for start, end, scene_fn, _ in SCENES:
            if start <= args.frame <= end:
                print(f"{scene_fn.__name__}, local frame {args.frame - start})")
                break
        else:
            print("unknown)")
        preview_frame(args.frame)
        return

    print("cpal 'Building Together' — Git Timeline Video")
    print(f"   Resolution: {WIDTH}x{HEIGHT} @ {FPS}fps")
    print(f"   Frames: {TOTAL_FRAMES} ({DURATION}s)")
    print()

    frames_dir = str(FRAMES_DIR)
    render_frames(frames_dir, TOTAL_FRAMES)

    print()
    audio = None if args.no_audio else args.audio
    assemble_video(frames_dir, args.output, audio_path=audio)

    print()
    print("Checking output...")
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries",
         "format=duration,size", "-show_entries",
         "stream=width,height,codec_name,r_frame_rate",
         "-of", "json", args.output],
        capture_output=True, text=True,
    )
    if probe.returncode == 0:
        import json
        info = json.loads(probe.stdout)
        print(f"   {json.dumps(info, indent=2)}")

    print()
    print(f"Done! Watch: {args.output}")


if __name__ == "__main__":
    main()
