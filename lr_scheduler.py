"""
Noam Learning Rate Scheduler
Reference: "Attention Is All You Need" (Vaswani et al., 2017)
           https://arxiv.org/abs/1706.03762

Formula:
    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


# ══════════════════════════════════════════════════════════════════════
#  NOAM SCHEDULER
# ══════════════════════════════════════════════════════════════════════

class NoamScheduler(LRScheduler):
    """
    Noam learning rate scheduler as described in "Attention Is All You Need".

    Applies a warm-up phase where LR increases linearly, followed by
    a decay phase where LR decreases proportional to the inverse square
    root of the step number.

    Args:
        optimizer    (torch.optim.Optimizer): Wrapped optimizer.
        d_model      (int) : Model dimensionality (embedding size).
        warmup_steps (int) : Number of warm-up steps before decay begins.
        last_epoch   (int) : The index of the last epoch. Default: -1.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:
        self.d_model       = d_model
        self.warmup_steps  = warmup_steps
        # Parent __init__ must come AFTER setting instance attrs
        # because it calls get_lr() internally
        super().__init__(optimizer, last_epoch)

    # ──────────────────────────────────────────────────────────────────

    def _get_lr_scale(self) -> float:
        """
        Compute the Noam scaling factor for the current step.

        step = last_epoch + 1  (avoids division by zero at step 0)

        scale = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        """
        step = self.last_epoch + 1          # 1-indexed step number
        scale = (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5)
        )
        return scale

    # ──────────────────────────────────────────────────────────────────

    def get_lr(self) -> list:
        """
        Compute learning rates for every param group.

        Multiplies each group's base_lr by the Noam scale factor.
        base_lrs are set by the parent class from the optimizer's
        initial param-group lr values.
        """
        scale = self._get_lr_scale()
        return [base_lr * scale for base_lr in self.base_lrs]


# ══════════════════════════════════════════════════════════════════════
#  HELPER — do NOT modify
# ══════════════════════════════════════════════════════════════════════

def get_lr_history(
    d_model: int,
    warmup_steps: int,
    total_steps: int,
) -> list:
    """
    Simulate the LR trajectory of NoamScheduler for `total_steps` steps.

    Args:
        d_model      (int): Model dimensionality.
        warmup_steps (int): Warm-up steps.
        total_steps  (int): Number of steps to simulate.

    Returns:
        list[float]: LR value at each step (length == total_steps).
    """
    dummy_model = torch.nn.Linear(1, 1)
    optimizer   = optim.Adam(dummy_model.parameters(), lr=1.0)
    scheduler   = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    history = []
    for _ in range(total_steps):
        history.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()

    return history


# ══════════════════════════════════════════════════════════════════════
#  QUICK VISUAL CHECK — run: python lr_scheduler.py
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    D_MODEL      = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS  = 20_000

    lrs = get_lr_history(D_MODEL, WARMUP_STEPS, TOTAL_STEPS)

    # Verify autograder criteria
    warmup_lrs = lrs[:WARMUP_STEPS]
    decay_lrs  = lrs[WARMUP_STEPS:]

    assert all(warmup_lrs[i] < warmup_lrs[i+1] for i in range(len(warmup_lrs)-1)), \
        "LR not monotonically increasing during warmup!"

    assert all(decay_lrs[i] > decay_lrs[i+1] for i in range(len(decay_lrs)-1)), \
        "LR not monotonically decreasing after warmup!"

    peak_idx = lrs.index(max(lrs))
    assert abs(peak_idx - WARMUP_STEPS) <= 10, \
        f"Peak at step {peak_idx}, expected near {WARMUP_STEPS}!"

    # Verify peak value matches closed-form formula
    expected_peak = (D_MODEL ** -0.5) * min(
        WARMUP_STEPS ** -0.5,
        WARMUP_STEPS * WARMUP_STEPS ** -1.5
    )
    actual_peak = max(lrs)
    assert abs(actual_peak - expected_peak) < 1e-8, \
        f"Peak LR {actual_peak} != expected {expected_peak}"

    print("All autograder checks passed ✓")
    print(f"Peak LR : {actual_peak:.6f}  at step ~{peak_idx}")
    print(f"LR @ step 1  : {lrs[0]:.8f}")
    print(f"LR @ warmup  : {lrs[WARMUP_STEPS-1]:.6f}")
    print(f"LR @ 20000   : {lrs[-1]:.6f}")

    plt.figure(figsize=(9, 4))
    plt.plot(lrs)
    plt.axvline(WARMUP_STEPS, color="red", linestyle="--", label=f"warmup={WARMUP_STEPS}")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.title(f"Noam LR Schedule  (d_model={D_MODEL})")
    plt.legend()
    plt.tight_layout()
    plt.savefig("noam_lr_curve.png", dpi=150)
    plt.show()
    print("Plot saved → noam_lr_curve.png")