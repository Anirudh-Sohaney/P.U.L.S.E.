"""Process-local request budget for the public signal API."""

from __future__ import annotations

import math
import threading
import time


class PublicRateLimiter:
    """A bounded per-client token bucket; one API process owns this state."""

    def __init__(self, capacity: float = 300, refill_per_second: float = 5):
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def reserve(self, client: str, cost: float) -> int:
        """Return zero when admitted, or seconds until the next affordable call."""
        now = time.monotonic()
        with self._lock:
            if client not in self._buckets and len(self._buckets) >= 4096:
                stale = now - self.capacity / self.refill_per_second
                self._buckets = {key: value for key, value in self._buckets.items()
                                 if value[1] > stale}
                if len(self._buckets) >= 4096:
                    oldest = min(self._buckets, key=lambda key: self._buckets[key][1])
                    self._buckets.pop(oldest)
            balance, last_seen = self._buckets.get(client, (self.capacity, now))
            balance = min(self.capacity, balance + max(0, now - last_seen) * self.refill_per_second)
            if balance < cost:
                self._buckets[client] = (balance, now)
                return max(1, math.ceil((cost - balance) / self.refill_per_second))
            self._buckets[client] = (balance - cost, now)
            return 0


def request_cost(path: str) -> int:
    if path.endswith('/history') or path.endswith('/latest'):
        return 5
    if path.endswith('/coverage') or path.endswith('/catalog'):
        return 3
    return 1
