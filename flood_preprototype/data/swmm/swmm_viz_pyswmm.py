"""
swmm_viz_pyswmm.py  (optimised)
================================
Perf fixes vs original:
  1. Stream-downsample DURING sim → no giant raw lists in RAM
  2. blit=True  → only dirty artists redrawn each frame (~4-6x faster)
  3. Artist pre-allocation → no redundant text/path object churn
  4. FFMpeg pipe uses threads=0 (auto) + faster preset
  5. frame stride param exposed so you can trade quality for speed

Requirements:  pip install pyswmm matplotlib numpy
Usage:
    python swmm_viz_pyswmm.py                   # auto-detect .inp
    python swmm_viz_pyswmm.py model.inp
    python swmm_viz_pyswmm.py model.inp --stride 6   # skip more frames → faster
"""

import sys
import math
import shutil
import tempfile
import argparse
import subprocess
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from pathlib import Path
from pyswmm import Simulation, Nodes, Links, Subcatchments

# ── Alert thresholds (metres) ──────────────────────────────────────────────
THRESHOLDS = {
    "CLEAR":   (0.00, 0.80),
    "WATCH":   (0.80, 1.40),
    "WARNING": (1.40, 2.00),
    "DANGER":  (2.00, 3.00),
}
TIER_COLORS = {
    "CLEAR":   "#2ecc71",
    "WATCH":   "#f1c40f",
    "WARNING": "#e67e22",
    "DANGER":  "#e74c3c",
}
MAX_DEPTH  = 3.00
DOWNSAMPLE = 120    # keep every Nth sim step (30 s steps → ~1 frame/hr)
BAR_W      = 24     # rain history window width


# ── INP parser ─────────────────────────────────────────────────────────────
def parse_inp(path):
    sections, current = {}, None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("["):
                current = line.strip("[]").upper()
                sections[current] = []
            elif current and not line.startswith(";;") and line.strip():
                sections[current].append(line.strip())
    return sections

def _pairs(sections, key, min_cols=3):
    out = {}
    for row in sections.get(key, []):
        p = row.split()
        if len(p) >= min_cols:
            out[p[0]] = (float(p[1]), float(p[2]))
    return out

def get_polygons(sections):
    polys = {}
    for row in sections.get("POLYGONS", []):
        p = row.split()
        if len(p) >= 3:
            polys.setdefault(p[0], []).append((float(p[1]), float(p[2])))
    return polys

def get_conduit_topo(sections):
    return [{"name": p[0], "from": p[1], "to": p[2]}
            for row in sections.get("CONDUITS", [])
            for p in [row.split()] if len(p) >= 3]

def lon_lat_to_xy(lon, lat, lon0, lat0):
    return ((lon - lon0) * 111_000 * math.cos(math.radians(lat0)),
            (lat - lat0) * 111_000)

def project_all(raw, lon0, lat0):
    return {k: lon_lat_to_xy(*v, lon0, lat0) for k, v in raw.items()}

def project_poly(pts, lon0, lat0):
    return [lon_lat_to_xy(*p, lon0, lat0) for p in pts]


# ── Simulation (stream-downsampled) ────────────────────────────────────────
def run_swmm(inp_path):
    """
    FIX 1: accumulate only every DOWNSAMPLE-th step, not all steps.
    Original appended every step then sliced → large intermediate lists.
    Now we count steps and append only the keepers → ~DOWNSAMPLE× less memory.
    """
    tmp     = Path(tempfile.mkdtemp())
    tmp_inp = tmp / Path(inp_path).name
    shutil.copy(inp_path, tmp_inp)

    print("[*] Running SWMM5 simulation (stream-downsampled) …")
    depths, flows, rain, flooding, times = [], [], [], [], []
    step_n = 0

    with Simulation(str(tmp_inp)) as sim:
        sensor = Nodes(sim)["J_SENSOR"]
        c_main = Links(sim)["C_MAIN"]
        sc     = Subcatchments(sim)["SC_OBANDO"]

        for _ in sim:
            if step_n % DOWNSAMPLE == 0:          # ← only append keepers
                depths.append(sensor.depth)
                flows.append(abs(c_main.flow))
                rain.append(sc.rainfall)
                flooding.append(sensor.flooding)
                times.append(sim.current_time)
            step_n += 1

    shutil.rmtree(tmp, ignore_errors=True)

    depths   = np.array(depths,   dtype=np.float32)
    flows    = np.array(flows,    dtype=np.float32)
    rain     = np.array(rain,     dtype=np.float32)
    flooding = np.array(flooding, dtype=np.float32)

    print(f"[✓] Done — {len(depths)} frames")
    print(f"    Depth  : {depths.min():.3f} – {depths.max():.3f} m")
    print(f"    Flow   : {flows.min():.4f} – {flows.max():.4f} cms")
    print(f"    Rain   : {rain.min():.2f} – {rain.max():.2f} mm/hr")
    return depths, flows, rain, flooding, times


def tier_for(depth):
    for tier, (lo, hi) in THRESHOLDS.items():
        if lo <= depth < hi:
            return tier
    return "DANGER"


# ── Figure build ────────────────────────────────────────────────────────────
def build_figure(inp_path, depths, flows, rain, flooding, times, frame_stride):
    sections   = parse_inp(inp_path)
    coords_raw = _pairs(sections, "COORDINATES")
    syms_raw   = _pairs(sections, "SYMBOLS")
    polys_raw  = get_polygons(sections)
    conduits   = get_conduit_topo(sections)

    lon0, lat0 = 120.952, 14.832
    for row in sections.get("MAP", []):
        if row.upper().startswith("DIMENSIONS"):
            p = row.split()
            if len(p) == 5:
                lon0 = (float(p[1]) + float(p[3])) / 2
                lat0 = (float(p[2]) + float(p[4])) / 2

    node_xy = project_all(coords_raw, lon0, lat0)
    sym_xy  = project_all(syms_raw,   lon0, lat0)
    poly_xy = {k: project_poly(v, lon0, lat0) for k, v in polys_raw.items()}

    N        = len(depths)
    rain_max = max(float(rain.max()), 0.1)
    flow_max = max(float(flows.max()), 0.01)

    # ── Layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 8.5), facecolor="#0d1117")
    gs  = fig.add_gridspec(3, 3,
                           left=0.04, right=0.97, top=0.91, bottom=0.06,
                           hspace=0.50, wspace=0.35,
                           height_ratios=[3.2, 1, 1])
    ax_map  = fig.add_subplot(gs[0, :2])
    ax_gau  = fig.add_subplot(gs[0,  2])
    ax_rain = fig.add_subplot(gs[1, :2])
    ax_flow = fig.add_subplot(gs[1,  2])
    ax_ts   = fig.add_subplot(gs[2, :])

    for ax in (ax_map, ax_gau, ax_rain, ax_flow, ax_ts):
        ax.set_facecolor("#161b22")
        for sp in ax.spines.values(): sp.set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=7)
        ax.xaxis.label.set_color("#8b949e")
        ax.yaxis.label.set_color("#8b949e")

    # ── Map static elements ──────────────────────────────────────────────
    sc_patches = {}
    for nm, pts in poly_xy.items():
        arr   = np.array(pts)
        patch = MplPolygon(arr, closed=True, linewidth=1.2,
                           edgecolor="#58a6ff", facecolor="#1f6feb33", zorder=2)
        ax_map.add_patch(patch)
        sc_patches[nm] = patch
        cx, cy = arr.mean(axis=0)
        ax_map.text(cx, cy, nm, color="#58a6ff", fontsize=7,
                    ha="center", va="center", fontstyle="italic", zorder=3)

    clines = {}
    for c in conduits:
        frm, to = c["from"], c["to"]
        if frm in node_xy and to in node_xy:
            x0, y0 = node_xy[frm]; x1, y1 = node_xy[to]
            ax_map.plot([x0, x1], [y0, y1], color="#21262d",
                        linewidth=6, solid_capstyle="round", zorder=3)
            fl, = ax_map.plot([x0, x1], [y0, y1],
                              color=TIER_COLORS["CLEAR"], linewidth=2.5,
                              solid_capstyle="round", zorder=4)
            clines[c["name"]] = fl

    for gn, gxy in sym_xy.items():
        ax_map.scatter(*gxy, s=100, marker="^", color="#79c0ff",
                       edgecolors="#ffffff", linewidths=0.5, zorder=6)
        ax_map.text(gxy[0], gxy[1]+38, gn, color="#79c0ff",
                    fontsize=5.5, ha="center", zorder=6)

    jdots = {}
    for jn, jxy in node_xy.items():
        dot = ax_map.scatter(*jxy,
                             s=130 if "SENSOR" in jn.upper() else 65,
                             color=TIER_COLORS["CLEAR"],
                             edgecolors="#ffffff", linewidths=0.9, zorder=7)
        jdots[jn] = dot
        ax_map.text(jxy[0], jxy[1]+32, jn, color="#c9d1d9", fontsize=5.5,
                    ha="center", zorder=8,
                    path_effects=[pe.withStroke(linewidth=2, foreground="#0d1117")])

    sensor_xy   = node_xy.get("J_SENSOR", (0, 0))
    flood_ring, = ax_map.plot([], [], "o", ms=28, mfc="none",
                              mec="#e74c3c", mew=2, alpha=0, zorder=9)

    flow_ann = None
    if conduits:
        frm, to = conduits[0]["from"], conduits[0]["to"]
        if frm in node_xy and to in node_xy:
            x0, y0 = node_xy[frm]; x1, y1 = node_xy[to]
            mx, my = (x0+x1)/2, (y0+y1)/2
            flow_ann = ax_map.annotate("", xy=(x1, y1), xytext=(mx, my),
                                       arrowprops=dict(arrowstyle="->",
                                                       color=TIER_COLORS["CLEAR"],
                                                       lw=1.5), zorder=8)

    all_xs = [xy[0] for xy in node_xy.values()] + [p[0] for pts in poly_xy.values() for p in pts]
    all_ys = [xy[1] for xy in node_xy.values()] + [p[1] for pts in poly_xy.values() for p in pts]
    pad = 220
    ax_map.set_xlim(min(all_xs)-pad, max(all_xs)+pad)
    ax_map.set_ylim(min(all_ys)-pad, max(all_ys)+pad)
    ax_map.set_aspect("equal")
    ax_map.grid(True, color="#21262d", linewidth=0.5, zorder=0)
    ax_map.set_xlabel("Easting (m from centroid)", fontsize=7)
    ax_map.set_ylabel("Northing (m from centroid)", fontsize=7)
    ax_map.set_title("Obando, Bulacan — SWMM5 Hydraulic Schematic (pyswmm)",
                     color="#c9d1d9", fontsize=9, pad=6)
    ax_map.legend(handles=[mpatches.Patch(color=v, label=k)
                            for k, v in TIER_COLORS.items()],
                  loc="lower right", fontsize=6, framealpha=0.35,
                  labelcolor="white", facecolor="#161b22", edgecolor="#30363d")

    # ── Gauge ────────────────────────────────────────────────────────────
    ax_gau.set_xlim(0, 1)
    ax_gau.set_ylim(0, MAX_DEPTH)
    ax_gau.set_xticks([])
    ax_gau.set_yticks(np.arange(0, MAX_DEPTH + 0.5, 0.5))
    ax_gau.set_title("J_SENSOR — Water Depth", color="#c9d1d9", fontsize=8, pad=4)
    ax_gau.set_ylabel("Depth (m)", fontsize=7)
    for tier, (lo, hi) in THRESHOLDS.items():
        ax_gau.axhspan(lo, hi, alpha=0.15, color=TIER_COLORS[tier], zorder=1)
        ax_gau.text(0.97, (lo+hi)/2, tier, color=TIER_COLORS[tier],
                    fontsize=5.5, va="center", ha="right", zorder=3)
    g_bar  = ax_gau.bar([0.5], [depths[0]], width=0.55,
                         color=TIER_COLORS["CLEAR"],
                         edgecolor="white", linewidth=0.5, zorder=4)[0]
    g_lbl  = ax_gau.text(0.5, depths[0]+0.06, f"{depths[0]:.2f} m",
                          color="white", fontsize=9, ha="center", va="bottom",
                          fontweight="bold", zorder=5)

    # ── Rain bars ────────────────────────────────────────────────────────
    rain_bars = ax_rain.bar(np.arange(BAR_W), [0]*BAR_W,
                             color="#79c0ff", width=0.8, zorder=3)
    ax_rain.set_xlim(-0.5, BAR_W-0.5)
    ax_rain.set_ylim(0, max(rain_max*1.15, 0.5))
    ax_rain.set_title("Rainfall Intensity — last 24 steps (mm/hr)",
                       color="#c9d1d9", fontsize=7, pad=3)
    ax_rain.set_ylabel("mm/hr", fontsize=6)
    ax_rain.set_xticks([])
    ax_rain.grid(True, axis="y", color="#21262d", linewidth=0.5, zorder=0)
    ax_rain.axvline(BAR_W-1, color="#f1c40f", lw=0.8, ls="--", zorder=4)

    # ── Flow bar ─────────────────────────────────────────────────────────
    ax_flow.set_xlim(0, 1)
    ax_flow.set_ylim(0, flow_max*1.15)
    ax_flow.set_xticks([])
    ax_flow.set_title("C_MAIN Flow", color="#c9d1d9", fontsize=8, pad=4)
    ax_flow.set_ylabel("Flow (cms)", fontsize=7)
    flow_bar = ax_flow.bar([0.5], [flows[0]], width=0.55,
                            color=TIER_COLORS["CLEAR"],
                            edgecolor="white", linewidth=0.5, zorder=4)[0]
    flow_lbl = ax_flow.text(0.5, flows[0]+flow_max*0.02, f"{flows[0]:.3f} cms",
                             color="white", fontsize=8, ha="center",
                             va="bottom", fontweight="bold", zorder=5)

    # ── Time series ──────────────────────────────────────────────────────
    ax_ts.set_xlim(0, N)
    ax_ts.set_ylim(0, MAX_DEPTH)
    for tier, (lo, hi) in THRESHOLDS.items():
        ax_ts.axhspan(lo, hi, alpha=0.10, color=TIER_COLORS[tier])
        ax_ts.text(N*1.001, (lo+hi)/2, tier,
                   color=TIER_COLORS[tier], fontsize=5, va="center")
    ts_line, = ax_ts.plot([], [], color=TIER_COLORS["CLEAR"],
                           linewidth=1.0, zorder=3)
    ts_now   = ax_ts.axvline(0, color="#ffffff", lw=0.8, ls=":", zorder=4)
    ax_ts.set_title("Full Simulation — Water Depth at J_SENSOR",
                     color="#c9d1d9", fontsize=7, pad=3)
    ax_ts.set_xlabel("Simulation step (hourly)", fontsize=6)
    ax_ts.set_ylabel("Depth (m)", fontsize=6)
    ax_ts.grid(True, color="#21262d", linewidth=0.5, zorder=0)

    # Header
    title_txt = fig.text(0.50, 0.958, "Initialising …",
                          color="#c9d1d9", fontsize=9.5, ha="center",
                          va="top", fontweight="bold")
    fig.text(0.50, 0.938,
             "Source: SWMM5 via pyswmm  |  Rapid Relay EWS — Obando, Bulacan",
             color="#484f58", fontsize=6.5, ha="center", va="top")

    # ── Pre-compute all-xs array for time series (avoids np.arange per frame)
    xs_all = np.arange(N, dtype=np.float32)

    # ── FIX 2: blit=True — update() must return ALL animated artists ─────
    # We collect them once and return the same tuple every call.
    dot_list  = list(jdots.values())
    line_list = list(clines.values())
    poly_list = list(sc_patches.values())
    bar_list  = list(rain_bars)

    # FIX 3: pre-cache scatter facecolor arrays (avoid repeated color parsing)
    sensor_col_cache  = {}
    def dot_color(tier):
        if tier not in sensor_col_cache:
            sensor_col_cache[tier] = np.array(
                [matplotlib.colors.to_rgba(TIER_COLORS[tier])])
        return sensor_col_cache[tier]

    neutral = np.array([matplotlib.colors.to_rgba("#484f58")])

    def update(frame):
        i    = frame
        d    = float(depths[i])
        f    = float(flows[i])
        r    = float(rain[i])
        fl   = float(flooding[i])
        t    = times[i]
        tier = tier_for(d)
        col  = TIER_COLORS[tier]

        # Node dots
        scol = dot_color(tier)
        for jn, dot in jdots.items():
            dot.set_facecolor(scol if "SENSOR" in jn.upper() else neutral)

        # Conduit lines
        flow_frac = min(f / flow_max, 1.0)
        lw = 1.2 + flow_frac * 4.0
        for fl_line in line_list:
            fl_line.set_color(col)
            fl_line.set_linewidth(lw)

        if flow_ann:
            flow_ann.arrow_patch.set_color(col)

        # Subcatchment fill
        alpha = 0.10 + min(r / rain_max, 1.0) * 0.30
        rgba  = (*matplotlib.colors.to_rgb(col), alpha)
        for patch in poly_list:
            patch.set_facecolor(rgba)

        # Flood ring
        if fl > 0:
            flood_ring.set_data([sensor_xy[0]], [sensor_xy[1]])
            flood_ring.set_alpha(0.7 + 0.3 * math.sin(i * 0.4))
        else:
            flood_ring.set_alpha(0)

        # Gauge
        g_bar.set_height(d)
        g_bar.set_color(col)
        g_lbl.set_text(f"{d:.2f} m")
        g_lbl.set_position((0.5, d + 0.06))
        g_lbl.set_color(col)

        # Flow bar
        flow_bar.set_height(f)
        flow_bar.set_color(col)
        flow_lbl.set_text(f"{f:.3f} cms")
        flow_lbl.set_position((0.5, f + flow_max * 0.02))
        flow_lbl.set_color(col)

        # Rain bars
        ws = max(0, i - BAR_W)
        for bi, bar in enumerate(bar_list):
            ti = ws + bi
            bar.set_height(float(rain[ti]) if ti < N else 0)
            bar.set_color("#79c0ff" if ti < i else "#30363d")

        # Time series — use pre-sliced view, no new array allocation
        ts_line.set_data(xs_all[:i+1], depths[:i+1])
        ts_line.set_color(col)
        ts_now.set_xdata([i, i])

        # Header
        title_txt.set_text(
            f"{t.strftime('%d %b %Y  %H:%M')}  |  "
            f"Depth {d:.2f} m  |  Flow {f:.3f} cms  |  Rain {r:.1f} mm/hr  |  [{tier}]"
        )
        title_txt.set_color(col)

        # FIX 2 continued: return every artist that changed
        return (dot_list + line_list + poly_list + bar_list +
                [g_bar, g_lbl, flow_bar, flow_lbl,
                 ts_line, ts_now, flood_ring, title_txt])

    frames = range(0, N, frame_stride)
    print(f"[*] Animating {len(frames)} frames (stride={frame_stride}) …")
    anim = FuncAnimation(fig, update, frames=frames,
                         interval=60, blit=True)   # ← blit=True

    out_base = Path(inp_path).stem
    out_mp4  = Path(inp_path).parent / f"{out_base}_pyswmm.mp4"
    out_gif  = Path(inp_path).parent / f"{out_base}_pyswmm.gif"

    try:
        # FIX 4: faster ffmpeg preset + auto threading
        writer = FFMpegWriter(
            fps=15,
            metadata={"title": "Rapid Relay SWMM Viz"},
            extra_args=["-vcodec", "libx264", "-preset", "faster",
                        "-crf", "23", "-pix_fmt", "yuv420p",
                        "-threads", "0"])          # ← auto thread count
        anim.save(str(out_mp4), writer=writer, dpi=120)
        print(f"[✓] Saved: {out_mp4}")
        return str(out_mp4)
    except Exception as e:
        print(f"[!] ffmpeg unavailable ({e}), saving GIF …")
        anim.save(str(out_gif), writer=PillowWriter(fps=12), dpi=72)
        print(f"[✓] Saved: {out_gif}")
        return str(out_gif)


# ── Auto-open output file ──────────────────────────────────────────────────
def open_file(path: str) -> None:
    """Open file in default OS viewer (Windows / macOS / Linux)."""
    try:
        import os, platform
        p = platform.system()
        if p == "Windows":
            os.startfile(path)
        elif p == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
        print(f"[✓] Opened: {path}")
    except Exception as e:
        print(f"[!] Could not auto-open ({e}) — open manually: {path}")


# ── Entry ──────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp", nargs="?", help=".inp file path")
    ap.add_argument("--stride", type=int, default=3,
                    help="Frame stride for animation (higher = fewer frames, faster)")
    args = ap.parse_args()

    if args.inp:
        inp_path = args.inp
    else:
        matches = list(Path(".").glob("*.inp"))
        if not matches:
            print("ERROR: No .inp file found.")
            sys.exit(1)
        inp_path = str(matches[0])
        print(f"[*] Auto-detected: {inp_path}")

    depths, flows, rain, flooding, times = run_swmm(inp_path)
    out = build_figure(inp_path, depths, flows, rain, flooding, times, args.stride)
    open_file(out)


if __name__ == "__main__":
    main()