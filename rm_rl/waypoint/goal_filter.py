"""Time-based waypoint hysteresis, independent of model and coordinate frame."""
import math


class GoalFilter:
    def __init__(self, confirm_s=1., min_hold_s=2.):
        if any(not math.isfinite(v) or v < 0 for v in (confirm_s, min_hold_s)):
            raise ValueError('hysteresis durations must be finite and nonnegative')
        self.confirm_s, self.min_hold_s = confirm_s, min_hold_s
        self.reset()

    def reset(self):
        self.current = self.pending = None
        self.since = self.pending_since = self.last_time = None
        self.reason = 'reset'

    def update(self, proposed, now, urgent=False):
        proposed = tuple(map(float, proposed))
        if len(proposed) != 2 or not all(map(math.isfinite, (*proposed, now))):
            raise ValueError('invalid goal or timestamp')
        if self.last_time is not None and now < self.last_time:
            self.reset()
        self.last_time = now
        if self.current is None or urgent:
            if self.current != proposed:
                self.since = now
            self.current, self.pending = proposed, None
            self.reason = 'urgent' if urgent else 'initial'
        elif proposed == self.current:
            self.pending = None
            self.reason = 'unchanged'
        else:
            if proposed != self.pending:
                self.pending, self.pending_since = proposed, now
            if now - self.pending_since + 1e-9 >= self.confirm_s and now - self.since + 1e-9 >= self.min_hold_s:
                self.current, self.since, self.pending = proposed, now, None
                self.reason = 'confirmed'
            else:
                self.reason = 'held'
        return self.current
