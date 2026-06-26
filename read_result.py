import os
import io
import pandas as pd
import numpy as np
from pathlib import Path

_ROTOR_COLS = ["CT_H", "CQ_H", "FOM", "Thrust", "Moment"]


def _read_rotor_sections(rotor_file):
    """Split a .rotor file into timestep lines and the header line.

    Returns (header_line, timestep_lines) where header_line is the column
    names row and timestep_lines are the numeric data rows (excluding the
    'Time averaged results' section).
    """
    with open(rotor_file) as f:
        raw = f.read()

    # Everything before "Time averaged results" is the timestep section
    # (pseudo-steady files don't have this section, so `before` == full text)
    before, _, _ = raw.partition("Time averaged results")
    lines = [l for l in before.splitlines() if l.strip()]

    # lines[0] is the description header, lines[1] is column names, rest is data
    header = lines[1]
    data_lines = lines[2:]
    return header, data_lines


def parse_rotor(rotor_file, avg_last_n=None):
    """Parse a VSPAERO .rotor file for global CT, CQ, FOM, etc.

    Parameters
    ----------
    avg_last_n : int or None
        - None or 1 : return the last timestep (pseudo-steady).
        - > 1       : mean of the last *n* timesteps (unsteady).
    """
    result = {}
    if not os.path.isfile(rotor_file):
        return result

    try:
        header, data_lines = _read_rotor_sections(rotor_file)
        csv_text = header + "\n" + "\n".join(data_lines)
        df = pd.read_csv(io.StringIO(csv_text), sep=r"\s+", engine="python")
        df.columns = df.columns.str.strip()

        if avg_last_n is not None and avg_last_n > 1:
            row = df.tail(avg_last_n)[_ROTOR_COLS].mean()
        else:
            row = df.iloc[-1]

        result["CT_H"]        = float(row["CT_H"])
        result["CQ_H"]        = float(row["CQ_H"])
        result["FOM_total"]       = float(row["FOM"])
        result["Thrust_total"]  = float(row["Thrust"])
        result["Moment_total"] = float(row["Moment"])

    except Exception as e:
        print(f"    WARNING: could not parse rotor file: {e}")
    return result


def _read_bref(lod_file):
    """Read the Bref_ reference length from a .lod file header."""
    with open(lod_file) as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("Bref_"):
                return float(stripped.split()[1])
    raise ValueError(f"Bref_ not found in {lod_file}")


def _read_lod_dataframe(lod_file):
    """Read a .lod file into a DataFrame, skipping the reference-value header.

    Handles both unsteady (first column ``Time``) and pseudo-steady
    (first column ``Iter``) formats.  A unified ``_step`` column is added
    so downstream code can group by timestep/iteration uniformly.
    """
    with open(lod_file) as f:
        raw = f.read()

    lines = raw.splitlines()
    # Find the column header: starts with "Time" or "Iter", contains "VortexSheet"
    header_idx = None
    for i, line in enumerate(lines):
        if not line.strip():  # Skip empty lines
            continue
        if "VortexSheet" in line and ("Time" in line or "Iter" in line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError(f"Could not find data header in {lod_file}")

    data_lines = [lines[header_idx]]
    for line in lines[header_idx + 1:]:
        if line.strip():
            data_lines.append(line)

    df = pd.read_csv(io.StringIO("\n".join(data_lines)), sep=r"\s+", engine="python")
    df.columns = df.columns.str.strip()

    # Unify the step column: unsteady uses "Time", pseudo-steady uses "Iter"
    if "Time" in df.columns:
        step_col = "Time"
    elif "Iter" in df.columns:
        step_col = "Iter"
    else:
        raise ValueError(f"Neither 'Time' nor 'Iter' column found in {lod_file}")

    df["_step"] = pd.to_numeric(df[step_col], errors="coerce")

    for col in ("roverR", "CT_h", "CQ_h", "dSpan", "Diameter", "VortexSheet", "Yavg"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_lod(lod_file, avg_last_n=None) -> dict[str, np.ndarray|int]:
    """Parse radial distributions from a VSPAERO .lod file.

    Works for both unsteady (``Time`` column) and pseudo-steady (``Iter``
    column) output.

    Warning! : r_norm(r/R) uses the Bref in the lod file to compute r/R

    Parameters
    ----------
    avg_last_n : int or None
        - None or 1 : use the last timestep/iteration only (pseudo-steady).
        - > 1       : average over the last *n* timesteps (unsteady).

    Returns
    -------
    dict with keys: r_norm, "CT_h", "CQ_h", "Thrust", "Moment", "FOM"  (lists, one entry per radial
    station, summed across all blades).


    """
    result = dict()
    if not os.path.isfile(lod_file):
        raise FileNotFoundError(f"lod_file {lod_file} not found from {Path.cwd()}")

    try:
        df = _read_lod_dataframe(lod_file)
        bref = _read_bref(lod_file)
        df["abs_Yavg"] = df["Yavg"].abs()

        # Select timesteps / iterations to average over
        steps = df["_step"].unique()
        if avg_last_n is not None and avg_last_n > 1:
            last_steps = steps[-avg_last_n:]
        else:
            last_steps = steps[-1:]
        subset = df[df["_step"].isin(last_steps)]

        # For each radial station: average aerodynamic loads across blades within each
        # average across steps.
        per_step = (
            subset.groupby(["_step", "roverR"])
            .agg(
                CT_h=("CT_h", "mean"),
                CQ_h=("CQ_h", "mean"),
                Thrust=("Thrust", "mean"),
                Moment=("Moment", "mean"),
                FOM=("FOM", "mean"),
                Cx = ("Cx", "mean"),
                Cy = ("Cy", "mean"),
                Cz = ("Cz", "mean"),
            )
            .reset_index()
        )
        per_blade = df[["Yavg", "Xavg", "Zavg", "Cy", "Cz", "dSpan", "Chord" ]][0:len(per_step)] #take geometry from first blade
        blade_num = df["VortexSheet"].max() # number of blades


        result["r_norm"] = (per_blade["Yavg"] / bref).to_numpy()
        result["CT_h"] = per_step["CT_h"].to_numpy()
        result["CQ_h"] = per_step["CQ_h"].to_numpy()
        result["Thrust"] = per_step["Thrust"].to_numpy()
        result["Moment"] = per_step["Moment"].to_numpy()
        result["FOM"] = per_step["FOM"].to_numpy()
        result["phase_angle"] = np.degrees(np.atan((-per_blade["Zavg"]/per_blade["Yavg"]).to_numpy())) # Y from root_LE to tip_LE x 90 degrees with y in aproximate positive chord direction
        result["polar_r"] = np.sqrt(per_blade["Yavg"].to_numpy()**2 + per_blade["Xavg"].to_numpy()**2)/bref
        result["Cx"] = per_step["Cx"].to_numpy()
        result["Cy"] = per_blade["Cy"].to_numpy()
        result["Cz"] = per_blade["Cz"].to_numpy()
        result["dSpan"] = per_blade["dSpan"].to_numpy()
        result["Chord"] = per_blade["Chord"].to_numpy()
        result["nB"] = blade_num

    except Exception as e:
        print(f"    WARNING: could not parse lod file: {e}")
    return result


def parse_results(case_dir, case_name, avg_last_n=None):
    """Parse output files from a completed VSPAERO case.

    Returns a dict with global coefficients (from .rotor) and radial
    distributions (from .lod).
    """
    rotor_file = os.path.join(case_dir, f"{case_name}.rotor.1")
    lod_file   = os.path.join(case_dir, f"{case_name}.lod")

    result = parse_rotor(rotor_file, avg_last_n=avg_last_n)
    result.update(parse_lod(lod_file, avg_last_n=avg_last_n))
    return result


if __name__ == "__main__":
    parse_lod("tangential_sweep_results/A02_amp51mm/A02_amp51mm.lod", avg_last_n=1)
