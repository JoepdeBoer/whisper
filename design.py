from dataclasses import dataclass
from enum import Enum

from numpy.typing import NDArray
import openvsp as vsp

@Enum
class CapType:
    NO_END_CAP = vsp.NO_END_CAP
    FLAT_END_CAP = vsp.FLAT_END_CAP
    ROUND_END_CAP = vsp.ROUND_END_CAP
    EDGE_END_CAP = vsp.EDGE_END_CAP
    SHARP_END_CAP = vsp.SHARP_END_CAP
    POINT_END_CAP = vsp.POINT_END_CAP
    ROUND_EXT_END_CAP_NONE = vsp.ROUND_EXT_END_CAP_NONE
    ROUND_EXT_END_CAP_LE = vsp.ROUND_EXT_END_CAP_LE
    ROUND_EXT_END_CAP_TE = vsp.ROUND_EXT_END_CAP_TE
    ROUND_EXT_END_CAP_BOTH = vsp.ROUND_EXT_END_CAP_BOTH


@Enum
class SectionType:
    XS_CIRCLE = vsp.XS_CIRCLE
    XS_ELLIPSE = vsp.XS_ELLIPSE
    XS_SUPER_ELLIPSE = vsp.XS_SUPER_ELLIPSE
    XS_ROUNDED_RECTANGLE = vsp.XS_ROUNDED_RECTANGLE
    XS_GENERAL_FUSE = vsp.XS_GENERAL_FUSE
    XS_FILE_FUSE = vsp.XS_FILE_FUSE
    XS_FOUR_SERIES = vsp.XS_FOUR_SERIES
    XS_SIX_SERIES = vsp.XS_SIX_SERIES
    XS_BICONVEX = vsp.XS_BICONVEX
    XS_WEDGE = vsp.XS_WEDGE
    XS_EDIT_CURVE = vsp.XS_EDIT_CURVE
    XS_FILE_AIRFOIL = vsp.XS_FILE_AIRFOIL
    XS_CST_AIRFOIL = vsp.XS_CST_AIRFOIL
    XS_VKT_AIRFOIL = vsp.XS_VKT_AIRFOIL
    XS_FOUR_DIGIT_MOD = vsp.XS_FOUR_DIGIT_MOD
    XS_FIVE_DIGIT = vsp.XS_FIVE_DIGIT
    XS_FIVE_DIGIT_MOD = vsp.XS_FIVE_DIGIT_MOD
    XS_ONE_SIX_SERIES = vsp.XS_ONE_SIX_SERIES
    XS_AC25_773 = vsp.XS_AC25_773


@dataclass
class design_variables():
    chord: NDArray[tuple[float, float, float]] # r/R, c/R, dc/dr
    twist: NDArray[tuple[float, float, float]] # r/R, deg, ddeg/dr
    axial: NDArray[tuple[float, float, float]] # r/R, axial/R, daxial/dr
    tangential: NDArray[tuple[float, float, float]] # r/R, tangential/R, tangential/dr`


@dataclass
class fixed_geom_params():
    R: float
    rhub_R: float
    thickness: NDArray[tuple[float, float, float]]  # r/R , t/c , dt/c/dr
    nB: int
    constructXc: float
    root_cap_type: CapType
    root_cap_length: float|None
    root_cap_offset: float|None
    root_cap_strength: float|None
    tip_cap_type: CapType
    tip_cap_length: float|None
    tip_cap_offset: float|None
    tip_cap_strength: float|None
    airfoil_type: SectionType


@dataclass
class tesselation_params:
    Num_U: int
    Num_W: int
    CapTess: int
    le_clustering: float # clustering is [0,1]
    te_clustering: float
    root_clustering: float
    tip_clustering: float
    min_le_te_panel_width: float
    max_growth: float




