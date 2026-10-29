"""Load Arena strategy modules in-process and trace the sentry's goal point."""
from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path


def _load(path, sdk_dir):
    sdk_dir = str(sdk_dir)
    if sdk_dir not in sys.path:
        sys.path.insert(0, sdk_dir)
    name = "arena_strategy_" + uuid.uuid4().hex
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load strategy: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "act", None)):
        raise RuntimeError(f"strategy has no act(): {path}")
    return module


class TracedSentryTeacher:
    """Teacher with a hook on drive() to expose the sentry goal.

    The hook is a read-only observer: it records the goal argument and then
    calls the original drive() with exactly the same arguments.
    """

    def __init__(self, path, sdk_dir):
        self.module = _load(path, sdk_dir)
        self.policy_errors = 0
        self.policy_error_messages = []
        if hasattr(self.module, 'decide'):
            original_decide = self.module.decide
            def observed(*args, **kwargs):
                try:
                    return original_decide(*args, **kwargs)
                except Exception as exc:
                    self.policy_errors += 1
                    if len(self.policy_error_messages) < 5:
                        self.policy_error_messages.append(str(exc))
                    raise
            self.module.decide = observed
        self.original_drive = getattr(self.module, "drive")
        self.last_goal = None
        self.module.drive = self._drive

    def _drive(self, robot, goal, obs, memory, tolerance=0.35):
        try:
            if robot.get("role") == "sentry" and str(robot.get("slot")) == "7":
                self.last_goal = (float(goal[0]), float(goal[1]))
        except (TypeError, ValueError, IndexError):
            pass
        return self.original_drive(robot, goal, obs, memory, tolerance)

    def act(self, observation, state):
        self.last_goal = None
        actions, new_state = self.module.act(observation, state)
        return actions, new_state


class StrategyRunner:
    def __init__(self, path, sdk_dir):
        self.module = _load(path, sdk_dir)

    def act(self, observation, state):
        return self.module.act(observation, state)
