"""Inclusive clinical thresholds with a numerical, not clinical, tolerance."""
import numpy as np

def ge_threshold(left, right):
    # The tolerance only absorbs binary floating-point roundoff at an exact boundary.
    return (left >= right) | np.isclose(left, right, rtol=0.0, atol=1e-12)
