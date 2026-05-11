#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Code from Juan Sebastián Salcedo-Gallo 
adjusted to compare markovian vs non-markovian dynamics

plot_two_panel_figure.py

Two-column figure (two amplitudes) with 4 rows:
  (0) Tail area line
  (1) Ringdown heatmap
  (2) Phase-V FFT heatmap
  (3) Floquet quasi-energies lines

Key features:
- Shared x-extent everywhere
- Shared y-limits for rows 0/1/2/3 across both columns
- Shared color scaling across both columns for ringdown + FFT maps
- One shared colorbar for ringdown (row 1), one for FFT (row 2), both placed in a dedicated right column
- Only left column shows y-axis labels/ticks
- Large Helvetica labels/ticks
- Amp annotation inside top row: r"$\Omega/2\pi$ = {amp} GHz"
- Row labels (a)-(d) at top-left of each row
- Column labels (i),(ii) inside each panel (black for rows 0&3, white for rows 1&2)

Run:
  python3 plot_two_panel_figure.py --npz data/two_tls_twoamp_data_myrun.npz --usetex
"""

import os
import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


def finite_minmax(x):
    a = np.asarray(x, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan, np.nan
    return float(a.min()), float(a.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True, help="NPZ from generate_two_panel_data.py")
    ap.add_argument("--outdir", type=str, default="plots")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--usetex", action="store_true")
    ap.add_argument("--cmap", type=str, default="inferno")

    # Big-figure defaults (edit here if you want even bigger)
    ap.add_argument("--font_base", type=float, default=60.0)
    ap.add_argument("--tick_size", type=float, default=60.0)
    ap.add_argument("--label_size", type=float, default=60.0)
    ap.add_argument("--annot_size", type=float, default=60.0)

    ap.add_argument("--row_label_size", type=float, default=72.0)  # (a)(b)(c)(d)
    ap.add_argument("--col_label_size", type=float, default=62.0)  # (i)(ii)

    ap.add_argument("--nbins_x", type=int, default=4)

    ap.add_argument("--tail_lw", type=float, default=5.5)
    ap.add_argument("--quasi_lw", type=float, default=7.0)

    ap.add_argument("--vline_lw", type=float, default=3.0)
    ap.add_argument("--vline_alpha", type=float, default=0.5)

    ap.add_argument("--drive_line_lw", type=float, default=3.0)
    ap.add_argument("--drive_line_alpha", type=float, default=0.6)

    ap.add_argument("--tail_pad_frac", type=float, default=0.06)
    ap.add_argument("--quasi_pad_frac", type=float, default=0.06)

    # Fine positioning of labels
    ap.add_argument("--row_label_x", type=float, default=-0.3)
    ap.add_argument("--row_label_y", type=float, default=1.0)
    ap.add_argument("--col_label_x", type=float, default=0.90)
    ap.add_argument("--col_label_y", type=float, default=0.92)

    # Figure size / spacing
    ap.add_argument("--fig_w", type=float, default=34.0)
    ap.add_argument("--fig_h", type=float, default=29.0)
    ap.add_argument("--cb_col_width", type=float, default=0.030)
    ap.add_argument("--wspace", type=float, default=0.10)
    ap.add_argument("--hspace", type=float, default=0.10)

    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    D = np.load(args.npz, allow_pickle=True)

    omega_d_vals = np.asarray(D["omega_d_vals"], float)
    t_us = np.asarray(D["t_us"], float)

    heat = np.asarray(D["heat"])                  # (2, Nt, n_freq)
    quasi = np.asarray(D["quasi"])                # (2, n_freq, 4) in GHz
    amps = np.asarray(D["amps"], float)           # (2,)
    tail_area_us_plot = np.asarray(D["tail_area_us_plot"], float)  # (2, n_freq)

    fft_map = np.asarray(D["fft_map"], float)     # (2, n_freq, n_fft)
    fft_freqs_mhz = np.asarray(D["fft_freqs_mhz"], float)

    omega1 = float(D["omega1"])
    omega2 = float(D["omega2"])
    t_drive_ns = float(D["t_drive_ns"])

    cutoff = float(D["gamma_bath"])

    fname = f"floquet_quasienergies_{cutoff}.png"

    if heat.shape[0] != 2:
        raise ValueError(f"Expected exactly 2 amplitudes, got heat.shape[0]={heat.shape[0]}")

    # ---------------- Style (Helvetica everywhere) ----------------
    plt.rcParams.update({
        "font.size": args.font_base,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "text.usetex": bool(args.usetex),
        "axes.linewidth": 2.0,
        "xtick.major.width": 1.8,
        "ytick.major.width": 1.8,
        "xtick.major.size": 9,
        "ytick.major.size": 9,
    })

    # ---------------- Shared limits ----------------
    x_min, x_max = float(omega_d_vals[0]), float(omega_d_vals[-1])

    # Tail y-lims shared across BOTH columns
    ty_min, ty_max = finite_minmax(tail_area_us_plot)
    if (not np.isfinite(ty_min)) or (not np.isfinite(ty_max)) or (ty_min == ty_max):
        ty_min, ty_max = 0.0, 1.0
    ty_pad = args.tail_pad_frac * (ty_max - ty_min if ty_max > ty_min else 1.0)
    ty_min = max(0.0, ty_min - ty_pad)
    ty_max = ty_max + ty_pad

    # Ringdown y-lims
    y_t_min, y_t_max = float(t_us[0]), float(t_us[-1])

    # Ringdown shared color scaling
    rd_vmin, rd_vmax = finite_minmax(heat)
    if (not np.isfinite(rd_vmin)) or (not np.isfinite(rd_vmax)) or (rd_vmin == rd_vmax):
        rd_vmin, rd_vmax = 0.0, 1.0

    # FFT y-lims
    y_fft_max = float(fft_freqs_mhz[-1])

    # FFT shared color scaling
    fft_vmin, fft_vmax = finite_minmax(fft_map)
    if (not np.isfinite(fft_vmin)) or (not np.isfinite(fft_vmax)) or (fft_vmin == fft_vmax):
        fft_vmin, fft_vmax = 0.0, 1.0

    # Quasi shared y-lims (GHz -> MHz)
    quasi_mhz = quasi * 1e3
    qy_min, qy_max = finite_minmax(quasi_mhz)
    if (not np.isfinite(qy_min)) or (not np.isfinite(qy_max)) or (qy_min == qy_max):
        qy_min, qy_max = -1.0, 1.0
    qy_pad = args.quasi_pad_frac * (qy_max - qy_min if qy_max > qy_min else 1.0)
    qy_min -= qy_pad
    qy_max += qy_pad

    # ---------------- Figure layout ----------------
    fig = plt.figure(figsize=(args.fig_w, args.fig_h))
    gs = gridspec.GridSpec(
        4, 3,
        height_ratios=[1.0, 1.5, 1.0, 1.0],
        width_ratios=[0.8, 0.8, args.cb_col_width],
        hspace=args.hspace,
        wspace=args.wspace
    )

    ax = np.empty((4, 2), dtype=object)
    for r in range(4):
        for c in range(2):
            sharex = ax[0, c] if r > 0 else None
            ax[r, c] = fig.add_subplot(gs[r, c], sharex=sharex)

    # Dedicated colorbar axes (do not steal subplot width)
    cax_ring = fig.add_subplot(gs[1, 2])
    cax_fft  = fig.add_subplot(gs[2, 2])
    fig.add_subplot(gs[0, 2]).axis("off")
    fig.add_subplot(gs[3, 2]).axis("off")

    def format_common(a):
        a.set_xlim(x_min, x_max)
        a.xaxis.set_major_locator(MaxNLocator(nbins=args.nbins_x))
        a.tick_params(axis="both", which="major", labelsize=args.tick_size, pad=12)

    def vlines(a, color):
        for w0 in (omega1, omega2):
            a.axvline(w0, ls="--", color=color, lw=args.vline_lw, alpha=args.vline_alpha)

    im_ring_ref = None
    im_fft_ref = None

    for col in range(2):
        A = float(amps[col])

        # Row 0: Tail area
        a0 = ax[0, col]
        a0.plot(omega_d_vals, tail_area_us_plot[col, :], lw=args.tail_lw)
        format_common(a0)
        vlines(a0, "k")
        a0.set_ylim(ty_min, ty_max)
        plt.setp(a0.get_xticklabels(), visible=False)

        a0.text(
            0.03, 0.86,
            rf"$\Omega/2\pi$ = {A:.3g} GHz",
            transform=a0.transAxes,
            fontsize=args.annot_size,
            ha="left", va="top",
            color="black"
        )

        if col == 0:
            a0.set_ylabel("Tail area (arb.)", fontsize=args.label_size, labelpad=16)
        else:
            a0.set_ylabel("")
            a0.tick_params(labelleft=False)

        # Row 1: Ringdown heatmap
        a1 = ax[1, col]
        im1 = a1.imshow(
            heat[col, :, :],
            extent=[x_min, x_max, y_t_min, y_t_max],
            origin="lower",
            aspect="auto",
            cmap=args.cmap,
            vmin=rd_vmin, vmax=rd_vmax
        )
        a1.axhline(t_drive_ns / 1e3, ls="--", color="w",
                   lw=args.drive_line_lw, alpha=args.drive_line_alpha)
        format_common(a1)
        vlines(a1, "w")
        a1.set_ylim(y_t_min, y_t_max)
        plt.setp(a1.get_xticklabels(), visible=False)

        if col == 0:
            a1.set_ylabel(r"Time ($\mu$s)", fontsize=args.label_size, labelpad=16)
        else:
            a1.set_ylabel("")
            a1.tick_params(labelleft=False)

        if im_ring_ref is None:
            im_ring_ref = im1

        # Row 2: FFT map
        a2 = ax[2, col]
        im2 = a2.imshow(
            fft_map[col, :, :].T,
            extent=[x_min, x_max, 0.0, y_fft_max],
            origin="lower",
            aspect="auto",
            cmap='Oranges',
            vmin=fft_vmin, vmax=fft_vmax
        )
        format_common(a2)
        vlines(a2, "k")
        a2.set_ylim(0.0, y_fft_max)
        plt.setp(a2.get_xticklabels(), visible=False)

        if col == 0:
            a2.set_ylabel(r"FFT Freq. (MHz)", fontsize=args.label_size, labelpad=20)
        else:
            a2.set_ylabel("")
            a2.tick_params(labelleft=False)

        if im_fft_ref is None:
            im_fft_ref = im2

        # Row 3: Quasi energies
        a3 = ax[3, col]

        handles = []
        labels = []

        for k in range(quasi_mhz.shape[2]):
            h, = a3.plot(
                omega_d_vals,
                quasi_mhz[col, :, k],
                lw=args.quasi_lw,
                label=rf"$Q_{{{k}}}$"
            )
            handles.append(h)
            labels.append(rf"$Q_{{{k}}}$")

        format_common(a3)
        vlines(a3, "k")
        a3.set_ylim(qy_min, qy_max)

        a3.set_xlabel("Drive Freq. (GHz)", fontsize=args.label_size, labelpad=16)

        if col == 0:
            a3.set_ylabel(r"$Q_\alpha$ (MHz)", fontsize=args.label_size, labelpad=-5)

            # ---- Legend ONLY for left panel ----
            a3.legend(
                handles, labels,
                loc="best",
                ncol=2,
                fontsize=40,
                frameon=False,
                handlelength=2.5,
                columnspacing=1.5,
                handletextpad=0.6,
                borderaxespad=0.3
            )
        else:
            a3.set_ylabel("")
            a3.tick_params(labelleft=False)


    # Shared colorbars
    cb1 = fig.colorbar(im_ring_ref, cax=cax_ring)
    cb1.ax.tick_params(labelsize=args.tick_size)
    cb1.set_label(r"$\langle \sigma^{+}\sigma^{-} \rangle$ (arb.)",
                  fontsize=args.label_size, labelpad=14)

    cb2 = fig.colorbar(im_fft_ref, cax=cax_fft)
    cb2.ax.tick_params(labelsize=args.tick_size)
    cb2.set_label(r"$|\mathrm{FFT}(\phi)|$ (arb.)",
                  fontsize=args.label_size, labelpad=14)

    # ------------------------------------------------------------
    # Panel labels: (a)-(d) per row, and (i)/(ii) per column panel
    # ------------------------------------------------------------
    row_labels = ["(a)", "(b)", "(c)", "(d)"]
    col_labels = ["(i)", "(ii)"]

    for r in range(4):
        # Row label on left column only
        ax[r, 0].text(
            args.row_label_x, args.row_label_y,
            row_labels[r],
            transform=ax[r, 0].transAxes,
            fontsize=args.row_label_size,
            fontweight="bold",
            ha="left", va="bottom",
            color="black"
        )

        for c in range(2):
            # black for top & bottom, white for middle rows
            # label_color = "black" if r in (0, 3) else "white"
            label_color = "black" if r in (0, 2, 3) else "white"
            ax[r, c].text(
                args.col_label_x, args.col_label_y,
                col_labels[c],
                transform=ax[r, c].transAxes,
                fontsize=args.col_label_size,
                fontweight="bold",
                ha="left", va="top",
                color=label_color
            )

    fig.subplots_adjust(left=0.125, right=0.90, top=0.96, bottom=0.08)
    
    outpath = os.path.join(args.outdir, fname)
    fig.savefig(outpath, dpi=args.dpi)
    plt.close(fig)
    print("✓ Figure saved →", outpath)


if __name__ == "__main__":
    main()
