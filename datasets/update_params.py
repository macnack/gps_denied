def get_translation(epoch, max_epoch=30, start=0.05, end=0.2):
    if epoch >= max_epoch:
        return end
    else:
        return start + (end - start) * (epoch / max_epoch)


def get_angle(epoch, max_epoch=30, start=3, end=20):
    if epoch >= max_epoch:
        return end
    else:
        return start + (end - start) * (epoch / max_epoch)


def get_scale_range(
    epoch,
    max_epoch=50,
    init_scale: tuple = (1.0, 1.0),
    goal_scale: tuple = (0.75, 1.25),
):
    start_min, start_max = init_scale
    end_min, end_max = goal_scale
    if epoch >= max_epoch:
        return (end_min, end_max)
    t = epoch / max_epoch
    min_scale = start_min + (end_min - start_min) * t
    max_scale = start_max + (end_max - start_max) * t
    return (min_scale, max_scale)


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


def parameter_ranges_check(parameters: dict):
    """
    Returns parameter ranges for homography warping.
    """
    assert (
        parameters["lower_sz"] > 0 and parameters["upper_sz"] > 0
    ), "Sizes must be positive."
    assert (
        parameters["min_scale"] > 0
        and parameters["max_scale"] >= parameters["min_scale"]
    ), "Scale must be positive."
    assert parameters["angle_range"] >= 0, "Angle range must be non-negative."
    assert parameters["projective_range"] >= 0, "Projective range must be non-negative."
    assert (
        parameters["translation_range"] >= 0.0
        and parameters["translation_range"] <= 1.0
    ), "Translation range must be in [0, 1]."

    return {
        "lower_sz": parameters["lower_sz"],
        "upper_sz": parameters["upper_sz"],
        "warp_pad": parameters["warp_pad"],
        "min_scale": parameters["min_scale"],
        "max_scale": parameters["max_scale"],
        "angle_range": parameters["angle_range"],
        "projective_range": parameters["projective_range"],
        "translation_range": parameters["translation_range"],
    }
