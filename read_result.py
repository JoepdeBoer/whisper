import os
import io
import pandas as pd


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
        stripped = line.strip()
        if ("VortexSheet" in stripped
                and (stripped.startswith("Time") or stripped.startswith("Iter"))):
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

    for col in ("roverR", "CT_h", "CQ_h", "dSpan", "Diameter", "VortexSheet"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_lod(lod_file, avg_last_n=None):
    """Parse radial distributions from a VSPAERO .lod file.

    Works for both unsteady (``Time`` column) and pseudo-steady (``Iter``
    column) output.

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
    result = dict(r_norm=[], dCT_dR=[], dCQ_dR=[])
    if not os.path.isfile(lod_file):
        return result

    try:
        df = _read_lod_dataframe(lod_file)

        # Select timesteps / iterations to average over
        steps = df["_step"].unique()
        if avg_last_n is not None and avg_last_n > 1:
            last_steps = steps[-avg_last_n:]
        else:
            last_steps = steps[-1:]
        subset = df[df["_step"].isin(last_steps)]

        # For each radial station: sum across blades within each step,
        # then average across steps.
        per_step = (subset.groupby(["_step", "roverR"])[[ "CT_h", "CQ_h", "Thrust", "Moment", "FOM"]]
                    .sum()
                    .reset_index())
        avg = per_step.groupby("roverR")[["CT_h", "CQ_h", "Thrust", "Moment", "FOM"]].mean()

        result["r_norm"]  = (avg.index  + (1-avg.index[-1])).tolist() # TODO remove hardcoded hub fraction/ sometimes exceeding 1.0 bug?
        result["CT_h"] = avg["CT_h"]
        result["CQ_h"] = avg["CQ_h"]
        result["Thrust"] = avg["Thrust"]
        result["Moment"] = avg["Moment"]
        result["FOM"] = avg["FOM"]

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
