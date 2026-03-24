import os
from io import StringIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd


def plot_sweep_summary(df, output_dir, diameter, rpm, vinf):
    """2x2 grid of CT, CQ, FOM, Thrust vs amplitude."""
    valid = df["CT_h"].notna()
    af = df["amplitude_frac_R"].values

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Propeller Tangential Sweep  (half-sine, 0 \u2192 R/4)\n"
        f"D={diameter*1000:.0f} mm  |  {rpm:.0f} RPM  |  "
        f"Vinf={vinf} m/s  |  Unsteady VLM",
        fontsize=11)
    for ax, col, title, color, fmt in [
        (axes[0, 0], "CT_H",       "Thrust Coefficient CT_H", "steelblue",  "o-"),
        (axes[0, 1], "CQ_H",       "Torque Coefficient CQ_H", "darkorange", "s-"),
        (axes[1, 0], "FOM_total",      "Figure of Merit",       "seagreen",   "^-"),
        (axes[1, 1], "Thrust_total", "Thrust  [N]",           "crimson",    "D-"),
    ]:
        if valid.any():
            ax.plot(af[valid], df[col][valid], fmt, color=color, ms=7, lw=1.8)
        ax.set_xlabel("Amplitude / R")
        ax.set_ylabel(col.replace("_N", " [N]"))
        ax.set_title(title)
        ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(output_dir, "sweep_summary.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"Plot \u2192 {p}")



def plot_tangential_shapes(amplitudes, output_dir, r_root_frac, r_tip_frac,
                           n_steps, radius):
    """Half-sine tangential curve shapes for each amplitude."""
    r_fracs = np.linspace(r_root_frac, r_tip_frac, 200)

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.plasma
    for i, A in enumerate(amplitudes):
        c = cmap(i / max(n_steps - 1, 1))
        offs = A * np.sin(np.pi * (r_fracs - r_root_frac) /
                          (r_tip_frac - r_root_frac))
        ax.plot(r_fracs, offs * 1000, color=c,
                label=f"A={A*1000:.1f}mm  (A/R={A/radius:.2f})")
    ax.set_xlabel("r / R")
    ax.set_ylabel("Tangential offset  [mm]")
    ax.set_title("Tangential Curve Shapes  (half-sine, root & tip = 0)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(output_dir, "tangential_shapes.png")
    plt.savefig(p, dpi=150); plt.close()
    print(f"Plot \u2192 {p}")

def plot_radial_distribution(df: pd.DataFrame, output_dir:str) ->None:

        # Build a colour map spanning all rows
        n = len(df)
        colours = cm.viridis(np.linspace(0.1, 0.9, n))

        # ---------- Figure 1: Thrust coefficient CT_h ----------
        fig_ct, ax_ct = plt.subplots(figsize=(8, 5))

        for i, row in df.iterrows():
            label = f"A/R = {row['amplitude_frac_R']:+.1f}"
            data = pd.read_csv(StringIO(row["CT_h"]), skipfooter=1, sep='\s+', skiprows=1, header=None)
            r = data[0]
            ct = data[1]
            ax_ct.plot(r, ct, color=colours[i], linewidth=1.8, label=label)

        ax_ct.set_xlabel("r / R  (–)", fontsize=13)
        ax_ct.set_ylabel(r"$c_{T}$  (–)", fontsize=13)
        ax_ct.set_title("Thrust Coefficient Distribution over Blade Span", fontsize=14)
        ax_ct.legend(title="Amplitude", fontsize=9, title_fontsize=10,
                     loc="upper left", framealpha=0.85)
        ax_ct.grid(True, linestyle="--", alpha=0.5)
        ax_ct.set_xlim(left=0)
        fig_ct.tight_layout()

        p = os.path.join(output_dir, "Thrust_distribution.png")
        plt.savefig(p, dpi=150)
        plt.close()

        # ---------- Figure 2: Torque coefficient CQ_h ----------
        fig_cq, ax_cq = plt.subplots(figsize=(8, 5))

        for i, row in df.iterrows():
            label = f"A/R = {row['amplitude_frac_R']:+.1f}"
            data = pd.read_csv(StringIO(row["CQ_h"]), skipfooter=1, sep='\s+', skiprows=1, header=None)
            r = data[0]
            cq = data[1]
            ax_cq.plot(r, cq, color=colours[i], linewidth=1.8, label=label)

        ax_cq.set_xlabel("r / R  (–)", fontsize=13)
        ax_cq.set_ylabel(r"$c_{Q}$  (–)", fontsize=13)
        ax_cq.set_title("Torque Coefficient Distribution over Blade Span", fontsize=14)
        ax_cq.legend(title="Amplitude", fontsize=9, title_fontsize=10,
                     loc="upper left", framealpha=0.85)
        ax_cq.grid(True, linestyle="--", alpha=0.5)
        ax_cq.set_xlim(left=0)
        fig_cq.tight_layout()

        p = os.path.join(output_dir, "Torque_distribution.png")
        plt.savefig(p, dpi=150)
        plt.close()

        return





def make_plots(df, output_dir, diameter, rpm, vinf):
    """Run all plot routines."""
    plot_sweep_summary(df, output_dir, diameter, rpm, vinf)
    plot_radial_distribution(df, output_dir)
    # plot_tangential_shapes(amplitudes, output_dir, r_root_frac, r_tip_frac,
    #                        n_steps, radius)



if __name__ == "__main__":
    path = os.path.join(os.path.dirname(__file__), "tangential_sweep_results", "sweep_summary.csv")
    df = pd.read_csv(path)
    make_plots(df=df,
               output_dir=os.path.join(os.path.dirname(__file__), "tangential_sweep_results"),
               diameter=0.508, rpm=5000, vinf=0)