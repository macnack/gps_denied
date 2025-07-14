def update_augmentation_difficulty(epoch, loss, dataset):
    new_ranges = {}
    # Thresholds and multipliers
    if loss < 10.0 and epoch >= 1:
        new_ranges = {
            "lower_sz": dataset.training_sz,
            "upper_sz": 120,
            "warp_pad": 0.4,
            "min_scale": 0.98,
            "max_scale": 1.02,
            "angle_range": 7,
            "projective_range": 0,
            "translation_range": 0.07,
        }
        return dataset.update_parameter_ranges(new_ranges)
    if loss < 5.0 and epoch > 20:
        new_ranges = {
            "lower_sz": dataset.training_sz,
            "upper_sz": 120,
            "warp_pad": 0.4,
            "min_scale": 0.95,
            "max_scale": 1.05,
            "angle_range": 10,
            "projective_range": 0,
            "translation_range": 0.1,
        }
        return dataset.update_parameter_ranges(new_ranges)
    return False
