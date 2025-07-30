def get_translation(epoch, max_epoch=30, start=0.05, end=0.2, step=0.01):
    if epoch >= max_epoch:
        value = end
    else:
        value = start + (end - start) * (epoch / max_epoch)

    if step is not None:
        value = round(value / step) * step

    return value


def get_angle(epoch, max_epoch=30, start=3, end=20, step=0.5):
    if epoch >= max_epoch:
        value = end
    else:
        value = start + (end - start) * (epoch / max_epoch)

    if step is not None:
        value = round(value / step) * step

    return value


def get_scale(
    epoch,
    max_epoch=50,
    init_scale: tuple = (1.0, 1.0),
    goal_scale: tuple = (0.75, 1.25),
    step: float = 0.05,
):
    def quantize(x, step):
        return round(x / step) * step

    start_min, start_max = init_scale
    end_min, end_max = goal_scale
    if epoch >= max_epoch:
        return (end_min, end_max)
    t = epoch / max_epoch
    min_scale = start_min + (end_min - start_min) * t
    max_scale = start_max + (end_max - start_max) * t
    return (quantize(min_scale, step), quantize(max_scale, step))


def update_parameter_ranges(
    epoch,
    start_translation: float = 0.05,
    end_translation: float = 0.2,
    translation_max_epoch: int = 30,
    start_angle: float = 3,
    end_angle: float = 20,
    angle_max_epoch: int = 30,
    init_scale: tuple = (1.0, 1.0),
    goal_scale: tuple = (0.75, 1.25),
    scale_max_epoch: int = 50,
):
    new_translation = get_translation(
        epoch,
        max_epoch=translation_max_epoch,
        start=start_translation,
        end=end_translation,
    )
    new_angle = get_angle(
        epoch, max_epoch=angle_max_epoch, start=start_angle, end=end_angle
    )
    new_scale = get_scale(
        epoch,
        max_epoch=scale_max_epoch,
        init_scale=init_scale,
        goal_scale=goal_scale,
    )
    parameter_ranges = {
        "min_scale": new_scale[0],
        "max_scale": new_scale[1],
        "angle_range": new_angle,
        "translation_range": new_translation,
    }
    return parameter_ranges


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
