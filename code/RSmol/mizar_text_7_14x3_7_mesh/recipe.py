"""Pure recipe mathematics, shared by runtime and dependency-free tests."""
import math


def validate_recipe(config):
    if min(config.world_size, config.micro_batch_size, config.gradient_accumulation_steps,
           config.max_optimizer_steps, config.scheduler_total_steps, config.save_every) <= 0:
        raise ValueError('Batch sizes and step counts must be positive')
    if not 0 <= config.warmup_steps < config.scheduler_total_steps:
        raise ValueError('Warmup must precede the end of the schedule')
    if not 0 < config.context_length <= 1024 or config.max_optimizer_steps > config.scheduler_total_steps:
        raise ValueError('Context must be in [1,1024] and updates within the scheduler horizon')
    if not (math.isfinite(config.max_lr) and math.isfinite(config.min_lr)
            and 0 < config.min_lr < config.max_lr):
        raise ValueError('Learning rates must be finite and 0 < min_lr < max_lr')
    if config.gate == 'FORMAL':
        expected = {'world_size': 8, 'context_length': 1024, 'max_optimizer_steps': 9244,
                    'scheduler_total_steps': 9244, 'warmup_steps': 463,
                    'steps_per_epoch': 9244, 'epochs': 1, 'save_every': 400,
                    'checkpoint_retention': 3, 'max_lr': 6e-4, 'min_lr': 6e-5,
                    'max_microbatches': None}
        if any(getattr(config, k) != v for k, v in expected.items()):
            raise ValueError('FORMAL requires the isolated 9244-step Mizar MeSH recipe')
        if (config.micro_batch_size, config.gradient_accumulation_steps) not in ((8, 16), (4, 32)):
            raise ValueError('FORMAL supports explicit micro8/GA16 or micro4/GA32 (GBS1024)')


def learning_rate(step, config):
    """Mizar completed-step formula: min at 0, max at warmup, min at horizon."""
    if not 0 <= step <= config.scheduler_total_steps:
        raise ValueError('Learning-rate update index outside schedule')
    if step < config.warmup_steps:
        return config.min_lr + (config.max_lr-config.min_lr) * step/max(1, config.warmup_steps)
    progress = (step-config.warmup_steps)/max(1, config.scheduler_total_steps-config.warmup_steps)
    return config.min_lr + 0.5*(config.max_lr-config.min_lr)*(1+math.cos(math.pi*progress))


def gradient_scale(world_size, global_window_tokens):
    """DDP averages rank gradients; each backward uses unnormalized CE SUM."""
    if world_size <= 0 or global_window_tokens <= 0:
        raise ValueError('World size and global token denominator must be positive')
    return world_size/global_window_tokens
