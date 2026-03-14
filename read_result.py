import os
import io
import pandas as pd


_COLS = ["CT", "CQ", "FOM", "Thrust", "Moment"]


def _read_rotor_sections(rotor_file):
    """Split a .rotor file into timestep lines and the header line.

    Returns (header_line, timestep_lines) where header_line is the column
    names row and timestep_lines are the numeric data rows (excluding the
    'Time averaged results' section).
    """
    with open(rotor_file) as f:
        raw = f.read()

    # Everything before "Time averaged results" is the timestep section
    before, _, _ = raw.partition("Time averaged results")
    lines = [l for l in before.splitlines() if l.strip()]

    # lines[0] is the description header, lines[1] is column names, rest is data
    header = lines[1]
    data_lines = lines[2:]
    return header, data_lines


def parse_rotor(rotor_file, avg_last_n=None):
    """Parse a VSPAERO .rotor file.

    Parameters
    ----------
    rotor_file : str
        Path to the .rotor.N file.
    avg_last_n : int or None
        - None or 1 : return the last timestep (use for pseudo-steady).
        - > 1       : return the mean of the last *n* timesteps
                       (use for unsteady, e.g. one full revolution).
    """
    result = dict(r_norm=[], r_m=[], dCT_dR=[])
    if not os.path.isfile(rotor_file):
        return result

    try:
        header, data_lines = _read_rotor_sections(rotor_file)
        csv_text = header + "\n" + "\n".join(data_lines)
        df = pd.read_csv(io.StringIO(csv_text), sep=r"\s+", engine="python")
        df.columns = df.columns.str.strip()

        if avg_last_n is not None and avg_last_n > 1:
            tail = df.tail(avg_last_n)
            row = tail[_COLS].mean()
        else:
            row = df.iloc[-1]

        result["CT"]        = float(row["CT"])
        result["CQ"]        = float(row["CQ"])
        result["FOM"]       = float(row["FOM"])
        result["Thrust_N"]  = float(row["Thrust"])
        result["Torque_Nm"] = float(row["Moment"])
    except Exception as e:
        print(f"    WARNING: could not parse rotor file: {e}")
    return result


def parse_results(case_dir, case_name, avg_last_n=None):
    """Parse output files from a completed VSPAERO case."""
    rotor_file = os.path.join(case_dir, f"{case_name}.rotor.1")
    return parse_rotor(rotor_file, avg_last_n=avg_last_n)