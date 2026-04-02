import os

from pathlib import Path
import numpy as np
import pytest
import openvsp as vsp
from pathlib import Path

from geom_utils import find_prop_geom, set_blade_curve_bezier
from vspaero_config import VSP_FILE

VSP_FILE = "nacapropeller-mod.vsp3"
write_loc = str(Path(__file__).parent / "testbezier.vsp3")

@pytest.fixture(scope="session", autouse=True)
def load_model() -> None:
    vsp.ClearVSPModel()
    vsp.ReadVSPFile(VSP_FILE)
    vsp.Update()

@pytest.fixture
def geom_id():
    return find_prop_geom()

@pytest.fixture
def radial_vec():
    return [0.2, 0.3, 0.4, 0.5, 0.6625, 0.83, 1]
@pytest.fixture
def valvec(radial_vec):
    len_values = len(radial_vec)
    len_derrivatives = len(radial_vec) * 2 - 2
    values = np.array(radial_vec)*0.2
    derivative_values = np.ones(len_derrivatives)*0.2
    vec = []
    for i in range(len_values):
        """Simple linear function order: value der der value der der value."""
        vec.append(values[i])
        if i < len_values - 1:
            vec.append(derivative_values[i*2])
            vec.append(derivative_values[i*2+1])
    return vec

def test_blade_pcurve(geom_id, radial_vec, valvec):
    # 2. PCurve no continuity
    set_blade_curve_bezier(geom_id, vsp.PROP_TANGENTIAL, valvec, radial_vec)
    set_blade_curve_bezier(geom_id, vsp.PROP_CHORD, valvec, radial_vec)
    set_blade_curve_bezier(geom_id, vsp.PROP_TWIST, valvec, radial_vec)

    # 4. save geometry immediately after Update()
    vsp.SetVSP3FileName(write_loc)
    vsp.WriteVSPFile(write_loc, vsp.SET_ALL)


