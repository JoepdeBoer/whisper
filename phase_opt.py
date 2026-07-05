# Optimise exponant k and offset c to minimise noise with a maximum 1 rad sweep
import csv
import os
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import numpy as np
from noise_analysis import run_noise_for_case
from read_result import parse_lod
from vspaero_config import AVG_LAST_N

MAX_SWEEP = .75 # radians
OUT_PUT_DIR = Path("noise_k_sweep_jul4_75")
LOD_PATH = Path("baseline/reverse_eng_TM.lod")


def phi(r, k):
    return -(r**k)*MAX_SWEEP + MAX_SWEEP

ksearch = np.linspace(1.58, 1.63, 5, endpoint=True)

def main():
    Path.mkdir(OUT_PUT_DIR, exist_ok=True)
    summary = []

    res = parse_lod(LOD_PATH, avg_last_n=AVG_LAST_N)
    r = np.array(res["polar_r"]) # radius along the x axis
    Fx = np.array(res["Fx"])
    Fy = np.array(res["Fy"])
    Fz = np.array(res["Fz"])

    for i, k in enumerate(ksearch):
        phase = phi(r, k)
        # if k >1.5 and k<1.8:
        #     plt.polar(phase,r)
        #     plt.show()

        name = f"noise_{MAX_SWEEP:.2f}_k{k:.3f}"
        row = run_noise_for_case(
            B =2,
            rR=r,
            Fx=Fx,
            phi0_base=np.degrees(phase),
            Fz=Fz,
            Fy=Fy, # TODO not implemented
            label=name,  # TODO not full path but file prefix only
            out_dir=OUT_PUT_DIR,
        )
        summary.append(row)

    # ── Comparison plot (only meaningful for ≥2 cases) ────────────────────
    if len(summary) >= 2:
        labels = [r["label"] for r in summary]
        x = np.arange(len(labels))

        fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(max(6, 2 * len(labels) + 2), 8))
        for metrics, ax, ylabel in [
            (["OSWLt", "OSWLd", "OSWLti", "OSWLdi"], ax_top, "OSWL [dB]"),
            (["OSWLtA", "OSWLdA", "OSWLtiA", "OSWLdiA"], ax_bot, "OSWLA [dBA]"),
        ]:
            for m in metrics:
                vals = [r[m] for r in summary]
                ax.plot(x, vals, "-o", label=m)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=15, ha="right")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)
        fig.suptitle("OSWL comparison across sweep cases")
        fig.tight_layout()
        path = os.path.join(OUT_PUT_DIR, "oswl_comparison.png")
        fig.savefig(path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Comparison plot → {path}")

    # ── Summary CSV ───────────────────────────────────────────────────────────
    if summary:
        csv_path = os.path.join(OUT_PUT_DIR, "noise_summary.csv")
        fieldnames = [
            "rank",
            "label",
            "OSWLA_total",
            "OSWLtA",
            "OSWLdA",
            "OSWLtiA",
            "OSWLdiA",
            "OSWLt",
            "OSWLd",
            "OSWLti",
            "OSWLdi",
        ]
        sorted_rows = sorted(
            summary, key=lambda r: r.get("OSWLA_total") or float("inf"), reverse=True
        )
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rank, row in enumerate(sorted_rows, start=1):
                writer.writerow(
                    {
                        "rank": rank,
                        **{
                            k: (
                                f"{row[k]:.2f}"
                                if isinstance(row.get(k), float) and np.isfinite(row[k])
                                else row.get(k, "")
                            )
                            for k in fieldnames
                            if k != "rank"
                        },
                    }
                )
        print(f"Summary CSV (loudest → quietest) → {csv_path}")

    print(f"\nDone. All noise results in '{OUT_PUT_DIR}/'.")


# def create_vsp3(baseline, k, max_sweep, name) -> None:



if __name__ == "__main__":
    main()


