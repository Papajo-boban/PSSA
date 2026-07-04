from bisect import bisect_left
from typing import List, Tuple


class ResourceTimeline:
    def __init__(self, periods: List[Tuple[int, int, int]]):
        self.segments: List[List[int]] = [
            [start, end, capacity]
            for start, end, capacity in sorted(periods)
            if start < end and capacity > 0
        ]

    def earliest_feasible_start(self, start: int, duration: int, amount: int) -> int:
        if duration <= 0:
            return start
        if not self.segments:
            return 10**9

        candidate = start
        while True:
            seg_idx = self._find_segment_index(candidate)
            if seg_idx >= len(self.segments):
                return 10**9

            seg_start, seg_end, seg_capacity = self.segments[seg_idx]
            if candidate < seg_start:
                candidate = seg_start
                seg_start, seg_end, seg_capacity = self.segments[seg_idx]

            if seg_capacity < amount:
                candidate = seg_end
                continue

            remaining = duration
            current_time = candidate
            idx = seg_idx

            while True:
                if idx >= len(self.segments):
                    return 10**9

                cur_start, cur_end, cur_capacity = self.segments[idx]
                if current_time < cur_start:
                    candidate = cur_start
                    break

                if cur_capacity < amount:
                    candidate = cur_end
                    break

                covered = cur_end - current_time
                if covered >= remaining:
                    return candidate

                remaining -= covered
                current_time = cur_end
                idx += 1
                if idx >= len(self.segments):
                    return 10**9
                next_start = self.segments[idx][0]
                if next_start > current_time:
                    candidate = next_start
                    break

    def commit(self, start: int, finish: int, amount: int) -> None:
        if amount <= 0 or finish <= start:
            return

        self._split_at(start)
        self._split_at(finish)

        idx = self._find_segment_index(start)
        while idx < len(self.segments):
            seg_start, seg_end, seg_capacity = self.segments[idx]
            if seg_start >= finish:
                break
            if seg_capacity < amount:
                raise ValueError("Committed infeasible resource usage.")
            self.segments[idx][2] -= amount
            idx += 1

        self._merge_adjacent()

    def _find_segment_index(self, time_value: int) -> int:
        lo = 0
        hi = len(self.segments)
        while lo < hi:
            mid = (lo + hi) // 2
            seg_start, seg_end, _ = self.segments[mid]
            if time_value < seg_start:
                hi = mid
            elif time_value >= seg_end:
                lo = mid + 1
            else:
                return mid
        return lo

    def _split_at(self, split_time: int) -> None:
        idx = self._find_segment_index(split_time)
        if idx >= len(self.segments):
            return
        seg_start, seg_end, seg_capacity = self.segments[idx]
        if split_time <= seg_start or split_time >= seg_end:
            return
        self.segments[idx:idx + 1] = [
            [seg_start, split_time, seg_capacity],
            [split_time, seg_end, seg_capacity],
        ]

    def _merge_adjacent(self) -> None:
        if not self.segments:
            return
        merged = [self.segments[0]]
        for seg_start, seg_end, seg_capacity in self.segments[1:]:
            last = merged[-1]
            if last[1] == seg_start and last[2] == seg_capacity:
                last[1] = seg_end
            else:
                merged.append([seg_start, seg_end, seg_capacity])
        self.segments = merged
