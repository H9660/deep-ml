def transformer_lr(step, d_model, warmup_steps, factor=1.0):
    """
    Learning rate at a single update step (step >= 1) following section 5.3 of the paper:
    factor * d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5).
    Returns a float.
    """
    res = 0
    for i in range(1, warmup_steps + 1):
        res = factor * (d_model ** (-0.5)) * min(step ** (-0.5), step * (warmup_steps ** (-1.5)))
    return res


def lr_schedule(num_steps, d_model, warmup_steps, factor=1.0):
    """
    List of learning rates for steps 1..num_steps, each rounded to 8 decimals.
    """
    import numpy as np
    list = [round(transformer_lr(i, d_model, warmup_steps, factor),8) for i in range(1, num_steps + 1)]
    return list
