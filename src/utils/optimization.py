from torch.optim.lr_scheduler import ConstantLR, ExponentialLR, SequentialLR
from transformers.optimization import get_constant_schedule_with_warmup

def get_constant_with_exponential_lr(
    optimizer,
    factor=0.1,
    total_iters_first=5,
    gamma=0.9,
    milestones=None
):
    if milestones is None:
        milestones = [2]

    scheduler1 = ConstantLR(optimizer, factor=factor, total_iters=total_iters_first)
    scheduler2 = ExponentialLR(optimizer, gamma=gamma)

    return SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=milestones)


def transformers_get_constant_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
):
    return get_constant_schedule_with_warmup(optimizer, num_warmup_steps)
