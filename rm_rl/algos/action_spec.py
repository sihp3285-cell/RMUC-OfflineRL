"""Action-space specification shared by every learner.

Two families are supported.

``continuous`` — the legacy 4-D layout ``(vx, vy, yaw_rate, fire)`` produced by
the ``velocity`` / ``goal`` dataset builds.  A single diagonal Gaussian covers
it.  Kept so old runs and checkpoints still load.

``tactical`` — the redesigned layout that matches how a real RoboMaster
autonomy stack is actually wired.  The decision layer does **not** drive the
gimbal or the trigger: auto-aim owns both loops at 100+ Hz, so a 1 Hz turret
*rate* is an aliased non-quantity and a 1 Hz *fire rate* is an outcome, not a
decision.  What the tactical layer genuinely decides is where to go, whom to
engage, and whether weapons are free::

    nav    (2, continuous)   sub-goal displacement in metres, scaled to [-1, 1]
    fire   (1, Bernoulli)    weapons-free gate for this second
    target (7, Categorical)  which enemy to prioritise; class 6 = "no target"

The target label is a **soft** distribution rather than a one-hot class.  The
referee log never records who was being shot at, so we recover it from the
muzzle bearing — and when several enemies sit inside the muzzle cone (55% of
firing seconds, measured) the assignment is genuinely ambiguous.  A soft label
represents that ambiguity honestly instead of guessing.

Discrete heads are also the direct fix for the failure mode that capped the
continuous policies: a Categorical is natively multi-modal, whereas an MSE /
single-Gaussian head regresses to the mean of a multi-modal target and emits
blurred, over-conservative actions.
"""
from __future__ import annotations

from dataclasses import dataclass

# 6 enemy slots (schema.MOBILE_TYPES order) + one "engage nobody" class
N_TARGET_CLASSES = 7
NO_TARGET = 6

# Target classes are schema.MOBILE_TYPES order: hero, engineer, infantry3,
# infantry4, aerial, sentry, no-target.  The aerial unit is a drone; it stays
# in the observation (it can shoot you) but it is not a valid ground target.
# Keep the action width at 7 so old checkpoints still load; the class is masked.
AERIAL_TARGET_SLOT = 4
TARGET_MASK = (1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True)
class ActionSpec:
    mode: str = "continuous"
    n_cont: int = 4          # continuous dims
    n_fire: int = 0          # Bernoulli dims (0 or 1)
    n_target: int = 0        # Categorical classes (0 = none)

    @property
    def dim(self) -> int:
        return self.n_cont + self.n_fire + self.n_target

    @property
    def is_tactical(self) -> bool:
        return self.mode == "tactical"

    @property
    def sl_cont(self) -> slice:
        return slice(0, self.n_cont)

    @property
    def sl_fire(self) -> slice:
        return slice(self.n_cont, self.n_cont + self.n_fire)

    @property
    def sl_target(self) -> slice:
        return slice(self.n_cont + self.n_fire, self.dim)


CONTINUOUS = ActionSpec("continuous", 4, 0, 0)
TACTICAL = ActionSpec("tactical", 2, 1, N_TARGET_CLASSES)

# names, for logging / deployment introspection
TACTICAL_NAMES = ("goal_dx", "goal_dy", "fire_gate") + tuple(
    f"target_{i}" for i in range(N_TARGET_CLASSES))
CONTINUOUS_NAMES = ("vx", "vy", "yaw_rate", "fire")


def get_spec(action_mode: str = "velocity", act_dim: int | None = None) -> ActionSpec:
    """Resolve the spec from a dataset's ``action_mode`` (see meta.json)."""
    if action_mode == "tactical":
        return TACTICAL
    return ActionSpec("continuous", int(act_dim or 4), 0, 0)


def spec_from_meta(meta: dict) -> ActionSpec:
    return get_spec(meta.get("action_mode", "velocity"),
                    meta.get("action_dim") or meta.get("expected_action_dim"))
