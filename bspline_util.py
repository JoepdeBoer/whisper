from scipy.interpolate import BSpline
import numpy as np
import matplotlib.pyplot as plt

def cubicbez_to_bspline(tvec: np.ndarray, valvec) -> BSpline:
    """
    Transfrom cubic bez to clamped cubic bspline.
    tvec is radial vector,
    valvec is param_value vector.
    """
    knots = []
    for i, r in enumerate(tvec):
        if i%3 == 0:
            if i == 0 or i == len(tvec)-1:
                knots.extend([(r-tvec[0])/(tvec[-1]-tvec[0])]*4)#mapping r to [0,1] edges clamped
            else:
                knots.extend([(r - tvec[0]) / (tvec[-1] - tvec[0])] * 3)  # mapping r to [0,1]


    controlpoints = np.column_stack((tvec, valvec))
    return BSpline(knots, controlpoints, k=3)


def sum_bsplines(y1:BSpline, y2:BSpline) -> BSpline:
    """
    Assumes same parametric space and degree.
    add y values of univariate bsplines y1,y2 and insert knots into x1.
    such that returned bsplines define the sum of two original functions
    """
    knots1 = set(y1.t)
    knots2 = set(y2.t)
    diff1 = knots1.difference(knots2)
    diff2 = knots2.difference(knots1)
    for k in diff1:
        y2 = y2.insert_knot(x=k  , m=3)
    for k in diff2:
        y1 = y1.insert_knot(x=k , m=3)
    summed_y = y1.c[:,1] + y2.c[:,1]
    new_c = np.column_stack((y1.c[:,0], summed_y))
    return BSpline(y1.t, new_c, k=3)

def bspline_to_cubicbez(bspline: BSpline) -> tuple[np.ndarray, np.ndarray]:
    "Takes bspline and returns (tvec, valvec)."
    tvec = bspline.c[:,0]
    valvec = bspline.c[:,1]
    return tvec, valvec

