
import math


def get_cosine_schedule_with_warmup_lr(
    current_step: int,
    base_lr: float, final_lr: float,
    num_warmup_steps: int, num_training_steps: int,
):
    assert current_step >= 0

    if current_step < num_warmup_steps:
        return base_lr * float(current_step) / float(num_warmup_steps)
    
    progress = min(
        1.0,
        float(current_step - num_warmup_steps) /
        max(1.0, float(num_training_steps - num_warmup_steps))
    )
    return final_lr + (base_lr - final_lr) * max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
