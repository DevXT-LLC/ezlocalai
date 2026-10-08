"""Bounded, process-local FIFO lanes for optional user/model worker affinity."""

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field


@dataclass
class UserLane:
    worker_id: str | None = None
    touched: float = field(default_factory=time.monotonic)
    users: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class UserAffinity:
    def __init__(self):
        self.lanes = {}

    @asynccontextmanager
    async def acquire(self, key, ttl=600, max_entries=10000, max_pending=100):
        now = time.monotonic()
        for old_key, lane in list(self.lanes.items()):
            if not lane.users and now - lane.touched >= ttl:
                del self.lanes[old_key]
        if key not in self.lanes and len(self.lanes) >= max_entries:
            idle = [
                (lane.touched, k) for k, lane in self.lanes.items() if not lane.users
            ]
            if not idle:
                raise OverflowError("User routing queue is full")
            del self.lanes[min(idle)[1]]
        lane = self.lanes.setdefault(key, UserLane())
        if lane.users >= max_pending:
            raise OverflowError("User routing queue is full")
        # Count queued requests too: TTL/pruning must never create a second lock
        # for a still-running lane. asyncio.Lock serves waiters in FIFO order.
        lane.users += 1
        try:
            async with lane.lock:
                yield lane
        finally:
            lane.users -= 1
            lane.touched = time.monotonic()
