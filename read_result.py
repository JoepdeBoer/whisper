import os
import pandas as pd

def parse_rotor(rotor_file):
    result = dict(r_norm=[], r_m=[], dCT_dR=[])
    if not os.path.isfile(rotor_file):
        return result
    try:
        df = pd.read_csv(rotor_file, sep=r"\s+", skiprows=1, engine="python")
        df.columns = df.columns.str.strip()
        row = df.iloc[-1]
        result["CT"]       = float(row["CT"])
        result["CQ"]       = float(row["CQ"])
        result["FOM"]      = float(row["FOM"])
        result["Thrust_N"]  = float(row["Thrust"])
        result["Torque_Nm"] = float(row["Moment"])
    except Exception as e:
        print(f"    WARNING: could not parse rotor file: {e}")
    return result


def parse_results(case_dir, case_name):
    """Parse output files from a completed VSPAERO case."""
    rotor_file = os.path.join(case_dir, f"{case_name}.rotor.1")
    rot = parse_rotor(rotor_file)
    return rot