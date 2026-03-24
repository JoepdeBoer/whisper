"""Tests for read_result parsers and plots using real VSPAERO output files.

Place .rotor.1 and .lod files in tangential_sweep_results/A00_amp0mm/ (or
adjust CASE_DIR below), then run:

    pytest test_parse_and_plot.py -v
"""
import os
import tempfile
import pytest
import pandas as pd

from read_result import parse_rotor, parse_lod, parse_results
from plots import plot_sweep_summary, plot_radial_distribution

# ── fixtures ────────────────────────────────────────────────────────────────

CASE_DIR  = os.path.join(os.path.curdir, "test_result_files")
CASE_NAME = "Unsteady"
ROTOR_FILE = os.path.join(CASE_DIR, f"{CASE_NAME}.rotor.1")
LOD_FILE   = os.path.join(CASE_DIR, f"{CASE_NAME}.lod")

needs_rotor = pytest.mark.skipif(
    not os.path.isfile(ROTOR_FILE),
    reason=f"{ROTOR_FILE} not found — provide test data")
needs_lod = pytest.mark.skipif(
    not os.path.isfile(LOD_FILE),
    reason=f"{LOD_FILE} not found — provide test data")


# ── parse_rotor ─────────────────────────────────────────────────────────────

@needs_rotor
class TestParseRotor:
    def test_last_timestep(self):
        res = parse_rotor(ROTOR_FILE)
        for key in ("CT_H", "CQ_H", "FOM", "Thrust", "Moment"):
            assert key in res, f"missing key: {key}"
            assert isinstance(res[key], float)

    def test_avg_last_n(self):
        res1 = parse_rotor(ROTOR_FILE, avg_last_n=1)
        res5 = parse_rotor(ROTOR_FILE, avg_last_n=5)
        # Both should return the same keys
        assert res1.keys() == res5.keys()
        # Averaged result should generally differ from single last step
        # (unless the simulation is perfectly converged)
        assert res5["CT"] != 0.0

    def test_avg_last_n_20(self):
        res = parse_rotor(ROTOR_FILE, avg_last_n=20)
        assert 0 < res["CT"] < 1, f"CT={res['CT']} out of expected range"
        assert res["Thrust_N"] > 0

    def test_missing_file_returns_empty(self):
        res = parse_rotor("nonexistent.rotor.1")
        assert res == {}


# ── parse_lod ───────────────────────────────────────────────────────────────

@needs_lod
class TestParseLod:
    def test_last_timestep(self):
        res = parse_lod(LOD_FILE)
        assert len(res["r_norm"]) > 0, "no radial stations parsed"
        assert len(res["r_norm"]) == len(res["dCT_dR"])
        assert len(res["r_norm"]) == len(res["dCQ_dR"])

    def test_r_norm_monotonic(self):
        res = parse_lod(LOD_FILE)
        r = res["r_norm"]
        assert r == sorted(r), "r_norm should be monotonically increasing"

    def test_r_norm_range(self):
        res = parse_lod(LOD_FILE)
        assert min(res["r_norm"]) > 0
        assert max(res["r_norm"]) <= 1.0

    def test_avg_last_n(self):
        res = parse_lod(LOD_FILE, avg_last_n=10)
        assert len(res["r_norm"]) > 0
        # dCT_dR should be positive for a thrust-producing propeller
        assert any(v > 0 for v in res["dCT_dR"])

    def test_missing_file_returns_empty_lists(self):
        res = parse_lod("nonexistent.lod")
        assert res == dict(r_norm=[], dCT_dR=[], dCQ_dR=[])


# ── parse_results (integration) ────────────────────────────────────────────

@needs_rotor
@needs_lod
class TestParseResults:
    def test_combined(self):
        res = parse_results(CASE_DIR, CASE_NAME, avg_last_n=20)
        # rotor keys
        for key in ("CT_H", "CQ_H", "FOM", "Thrust_total", "Moment_total"):
            assert key in res
        # lod keys
        assert len(res["r_norm"]) > 0
        assert len(res["CT_h"]) == len(res["r_norm"])
        assert len(res["CQ_h"]) == len(res["r_norm"])
        assert len(res["Moment"]) == len(res["r_norm"])
        assert len(res["Thrust"]) == len(res["r_norm"])
        assert len(res["FOM"]) == len(res["r_norm"])


# ── plots ───────────────────────────────────────────────────────────────────

@needs_rotor
@needs_lod
class TestPlots:
    def _make_summary_df(self):
        """Build a minimal sweep DataFrame from the test case."""
        res = parse_results(CASE_DIR, CASE_NAME, avg_last_n=20)
        return pd.DataFrame([dict(
            step=0, amplitude_m=0.0, amplitude_frac_R=0.0,
            CT=res.get("CT"), CQ=res.get("CQ"),
            Thrust_N=res.get("Thrust_N"), Torque_Nm=res.get("Torque_Nm"),
            FOM=res.get("FOM"),
        )])

    def _write_radial_csv(self, tmpdir):
        """Write a radial_distribution.csv so the plot function can find it."""
        res = parse_results(CASE_DIR, CASE_NAME, avg_last_n=20)
        case_subdir = os.path.join(tmpdir, "A00_amp0.0mm")
        os.makedirs(case_subdir, exist_ok=True)
        pd.DataFrame({
            "r_norm": res["r_norm"],
            "dCT_dR": res["dCT_dR"],
            "dCQ_dR": res["dCQ_dR"],
        }).to_csv(os.path.join(case_subdir, "radial_distribution.csv"),
                  index=False)

    def test_sweep_summary_creates_png(self):
        df = self._make_summary_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_sweep_summary(df, tmpdir, diameter=0.508, rpm=5000, vinf=0.0)
            assert os.path.isfile(os.path.join(tmpdir, "sweep_summary.png"))

    def test_radial_thrust_creates_png(self):
        df = self._make_summary_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_radial_csv(tmpdir)
            plot_radial_distribution(
                df, tmpdir, amplitude_frac=0.6,
                col="dCT_dR", ylabel="dCT / d(r/R)",
                title="test thrust", filename="thrust_dist.png")
            assert os.path.isfile(os.path.join(tmpdir, "thrust_dist.png"))

    def test_radial_torque_creates_png(self):
        df = self._make_summary_df()
        with tempfile.TemporaryDirectory() as tmpdir:
            self._write_radial_csv(tmpdir)
            plot_radial_distribution(
                df, tmpdir, amplitude_frac=0.6,
                col="dCQ_dR", ylabel="dCQ / d(r/R)",
                title="test torque", filename="torque_dist.png")
            assert os.path.isfile(os.path.join(tmpdir, "torque_dist.png"))