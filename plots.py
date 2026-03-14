import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_sweep_summary(df, output_dir, diameter, rpm, vinf):
    """2x2 grid of CT, CQ, FOM, Thrust vs amplitude."""
    valid = df["CT"].notna()
    af = df["amplitude_frac_R"].values

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle(
        f"Propeller Tangential Sweep  (half-sine, 0 \u2192 R/4)\n"
        f"D={diameter*1000:.0f} mm  |  {rpm:.0f} RPM  |  "
        f"Vinf={vinf} m/s  |  Unsteady VLM",
        fontsize=11)
    for ax, col, title, color, fmt in [
        (axes[0, 0], "CT",       "Thrust Coefficient CT", "steelblue",  "o-"),
        (axes[0, 1], "CQ",       "Torque Coefficient CQ", "darkorange", "s-"),
        (axes[1, 0], "FOM",      "Figure of Merit",       "seagreen",   "^-"),
        (axes[1, 1], "Thrust_N", "Thrust  [N]",           "crimson",    "D-"),
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


def plot_radial_thrust(df, output_dir, amplitude_frac):
    """Radial thrust distribution for each valid amplitude."""
    valid = df["CT"].notna()

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.cm.viridis
    for _, row in df[valid].iterrows():
        f = os.path.join(output_dir,
                f"A{int(row['step']):02d}_amp{row['amplitude_m']*1000:.1f}mm",
                "thrust_distribution.csv")
        if os.path.exists(f):
            d = pd.read_csv(f)
            c = cmap(row["amplitude_frac_R"] / (amplitude_frac + 1e-9))
            ax.plot(d["r_norm"], d["dCT_dR"], color=c,
                    label=f"A/R={row['amplitude_frac_R']:.2f}")
    ax.set_xlabel("r / R")
    ax.set_ylabel("dCT / dR")
    ax.set_title("Radial Thrust Distribution \u2014 tangential amplitude sweep")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    p = os.path.join(output_dir, "thrust_distribution_sweep.png")
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


def make_plots(df, amplitudes, *, output_dir, diameter, rpm, vinf,
               amplitude_frac, r_root_frac, r_tip_frac, n_steps, radius):
    """Run all three plot routines."""
    plot_sweep_summary(df, output_dir, diameter, rpm, vinf)
    plot_radial_thrust(df, output_dir, amplitude_frac)
    plot_tangential_shapes(amplitudes, output_dir, r_root_frac, r_tip_frac,
                           n_steps, radius)