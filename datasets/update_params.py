def default_parameter_ranges(training_sz: int):
    """
    Returns default parameter ranges for homography warping.
    """
    assert training_sz > 0, "Training size must be positive."
    parameter_ranges = {
        "lower_sz": training_sz,
        "upper_sz": training_sz,
        "warp_pad": 0.4,
        "min_scale": 1.0,
        "max_scale": 1.0,
        "angle_range": 3,
        "projective_range": 0,
        "translation_range": 0.1,
    }
    return parameter_ranges


def parameter_ranges_check(parmaters: dict):
    """
    Returns parameter ranges for homography warping.
    """
    assert (
        parmaters["lower_sz"] > 0 and parmaters["upper_sz"] > 0
    ), "Sizes must be positive."
    assert (
        parmaters["min_scale"] > 0 and parmaters["max_scale"] >= parmaters["min_scale"]
    ), "Scale must be positive."
    assert parmaters["angle_range"] >= 0, "Angle range must be non-negative."
    assert parmaters["projective_range"] >= 0, "Projective range must be non-negative."
    assert (
        parmaters["translation_range"] >= 0.0 and parmaters["translation_range"] <= 1.0
    ), "Translation range must be in [0, 1]."

    return {
        "lower_sz": parmaters["lower_sz"],
        "upper_sz": parmaters["upper_sz"],
        "warp_pad": parmaters["warp_pad"],
        "min_scale": parmaters["min_scale"],
        "max_scale": parmaters["max_scale"],
        "angle_range": parmaters["angle_range"],
        "projective_range": parmaters["projective_range"],
        "translation_range": parmaters["translation_range"],
    }
