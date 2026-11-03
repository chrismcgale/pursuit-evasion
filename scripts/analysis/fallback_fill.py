"""Hand the defender law's fallback regime to another controller.

Since 2026-11-03 the shipped defender gate hands nothing over, so analysis that
needs the old handover builds it here. ``FallbackFill(law, fill)`` is exactly
the old gated defender: on each agent-tick where ``law.plan()`` flags its
fallback, the agent flies ``fill``'s action; otherwise the law's. Both actions
are clipped to [-1, 1], as GatedController did.
"""
import numpy as np


class FallbackFill:
    def __init__(self, law, fill, name=None):
        self.law, self.fill = law, fill
        self.name = name or f"{getattr(law, 'name', 'law')}+{getattr(fill, 'name', 'fill')}"
        self.fill_ticks = 0
        self.ticks = 0

    def reset(self):
        self.law.reset()
        self.fill.reset()

    def act(self, view):
        a, fb = self.law.plan(view)
        a = np.clip(np.asarray(a, dtype=np.float64).reshape(-1, 3), -1, 1)
        if np.any(fb):
            f = np.clip(np.asarray(self.fill.act(view), dtype=np.float64).reshape(-1, 3), -1, 1)
            for i, flag in enumerate(fb):
                if flag and view.self_alive[i]:
                    a[i] = f[i]
                    self.fill_ticks += 1
        self.ticks += int(np.sum(view.self_alive))
        return a.reshape(-1)
