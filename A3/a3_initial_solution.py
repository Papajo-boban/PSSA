from dataclasses import dataclass
import heapq
import importlib.util
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Set, Tuple

from pms_instance import PMSInstance
from resource_timeline import ResourceTimeline


@dataclass
class CandidateMove:
    score: Tuple[int, ...]
    job_id: int
    machine_id: int
    insert_pos: int
    start: int
    finish: int


@dataclass
class ScheduleState:
    machine_sequences: List[List[int]]
    machine_assignment: List[int]
    start_times: List[int]
    finish_times: List[int]
    machine_ready_time: List[int]
    previous_job_by_machine: List[Optional[int]]
    resource_usage: List[List[Tuple[int, int, int, int]]]
    scheduled_jobs: Set[int]
    insertion_order: List[int]


@dataclass
class ScheduleEvaluation:
    feasible: bool
    infeasibility_score: int
    objective: int
    tardiness: int
    makespan: int
    solution: Dict
    start_times: List[int]
    finish_times: List[int]
    machine_assignment: List[int]
    violating_jobs: Set[int]
    violation_summary: Dict[str, int]
    overload_events: List[Dict]


@dataclass
class BeamNode:
    state: ScheduleState
    unscheduled_jobs: Set[int]
    remaining_predecessors: List[int]
    ready_jobs: Set[int]
    resource_timelines: List[ResourceTimeline]


class InitialSolutionBuilder:
    def __init__(
        self,
        instance: PMSInstance,
        frontier_size: int = 12,
        rollback_size: int = 8,
        max_rollbacks: int = 200,
        max_shift_positions: int = 3,
        random_seed: int = 0,
    ):
        self.instance = instance
        self.frontier_size = frontier_size
        self.rollback_size = rollback_size
        self.max_rollbacks = max_rollbacks
        self.max_shift_positions = max_shift_positions
        self.rng = random.Random(random_seed)

    def initial_feasible_solution(self) -> Dict:
        return self.initial_feasible_solution_with_deadline(None)

    def initial_feasible_solution_with_deadline(self, deadline: Optional[float]) -> Dict:
        return self.greedy_list_schedule(deadline)

    def greedy_list_schedule(self, deadline: Optional[float]) -> Dict:
        machine_sequences = [[] for _ in range(self.instance.n_machines + 1)]
        machine_ready_time = [0] * (self.instance.n_machines + 1)
        machine_last_job = [None] * (self.instance.n_machines + 1)
        start_times = [0] * self.instance.n_jobs
        finish_times = [0] * self.instance.n_jobs
        machine_assignment = [0] * self.instance.n_jobs
        scheduled_jobs: Set[int] = set()
        remaining_predecessors = [len(preds) for preds in self.instance.predecessor_indices]
        ready_heap: List[Tuple[Tuple[int, int, int, int, int], int]] = []
        resource_timelines = self.instance._build_resource_timelines()

        for job_id, count in enumerate(remaining_predecessors):
            if count == 0:
                heapq.heappush(ready_heap, (self.instance.job_priority_key(job_id), job_id))

        while ready_heap:
            if self._time_exceeded(deadline):
                raise RuntimeError("Timed out while building a feasible initial solution.")

            _, job_id = heapq.heappop(ready_heap)
            if job_id in scheduled_jobs:
                continue

            predecessor_finish = max(
                (finish_times[pred_id] for pred_id in self.instance.predecessor_indices[job_id]),
                default=0,
            )

            best_move: Optional[CandidateMove] = None
            for machine_id in self.instance.eligible[job_id]:
                if machine_last_job[machine_id] is None:
                    earliest_start = max(predecessor_finish, self.instance.initial_setup[job_id])
                else:
                    earliest_start = max(
                        predecessor_finish,
                        machine_ready_time[machine_id] + self.instance.setup[job_id][machine_last_job[machine_id]],
                    )

                feasible_start = earliest_start
                while True:
                    updated_start = feasible_start
                    for resource_id, required_amount in self.instance.job_resource_requirements[job_id]:
                        candidate_start = resource_timelines[resource_id].earliest_feasible_start(
                            updated_start,
                            self.instance.processing[job_id],
                            required_amount,
                        )
                        if candidate_start >= 10**9:
                            updated_start = 10**9
                            break
                        updated_start = max(updated_start, candidate_start)
                    if updated_start >= 10**9:
                        feasible_start = 10**9
                        break
                    if updated_start == feasible_start:
                        break
                    feasible_start = updated_start
                if feasible_start >= 10**9:
                    continue

                finish = feasible_start + self.instance.processing[job_id]
                score = (
                    finish,
                    len(self.instance.eligible[job_id]),
                    self.instance.resource_weights[job_id],
                    job_id,
                )
                move = CandidateMove(score, job_id, machine_id, len(machine_sequences[machine_id]), feasible_start, finish)
                if best_move is None or move.score < best_move.score:
                    best_move = move

            if best_move is None:
                raise RuntimeError(f"Could not place ready job {job_id}.")

            machine_id = best_move.machine_id
            machine_sequences[machine_id].append(job_id)
            machine_assignment[job_id] = machine_id
            start_times[job_id] = best_move.start
            finish_times[job_id] = best_move.finish
            machine_ready_time[machine_id] = best_move.finish
            machine_last_job[machine_id] = job_id
            scheduled_jobs.add(job_id)
            for resource_id, required_amount in self.instance.job_resource_requirements[job_id]:
                resource_timelines[resource_id].commit(best_move.start, best_move.finish, required_amount)

            for successor_id in self.instance.successor_indices[job_id]:
                remaining_predecessors[successor_id] -= 1
                if remaining_predecessors[successor_id] == 0:
                    heapq.heappush(ready_heap, (self.instance.job_priority_key(successor_id), successor_id))

        if len(scheduled_jobs) != self.instance.n_jobs:
            raise RuntimeError("Could not build a feasible initial solution.")

        objective, tardiness, makespan, feasible, solution, _, _, _ = self.instance.decode_sequences(machine_sequences)
        if not feasible:
            raise RuntimeError("Constructed schedule failed final feasibility decoding.")

        return {
            "objective": objective,
            "tardiness": tardiness,
            "makespan": makespan,
            "solution": solution,
            "job_sequences_by_machine": machine_sequences,
        }

    def build_large_instance_feasible_solution(self, deadline: Optional[float]) -> Dict:
        state = self._empty_state()
        unscheduled_jobs = set(range(self.instance.n_jobs))
        remaining_predecessors = [len(preds) for preds in self.instance.predecessor_indices]
        ready_jobs = {job_id for job_id, count in enumerate(remaining_predecessors) if count == 0}
        resource_timelines = self.instance._build_resource_timelines()
        rollback_count = 0

        while unscheduled_jobs:
            if self._time_exceeded(deadline):
                raise RuntimeError("Timed out while building a feasible initial solution.")

            remaining_resource_demand = self._remaining_resource_demand(unscheduled_jobs)
            pressured_window = self._find_windowed_resource_pressure(
                state,
                unscheduled_jobs,
                resource_timelines,
            )
            if pressured_window is not None and rollback_count < self.max_rollbacks and state.insertion_order:
                removed_jobs = self._rollback_resource_pressure_state(
                    state,
                    unscheduled_jobs,
                    resource_id=pressured_window[0],
                    window_start=pressured_window[1],
                    window_end=pressured_window[2],
                )
                if removed_jobs:
                    rollback_count += 1
                    remaining_predecessors, ready_jobs = self._recompute_frontier(unscheduled_jobs, state.scheduled_jobs)
                    resource_timelines = self._build_resource_timelines_from_state(state)
                    continue

            dead_resource = self._find_aggregate_resource_impossibility(remaining_resource_demand, resource_timelines)
            if dead_resource is not None:
                if rollback_count >= self.max_rollbacks or not state.insertion_order:
                    resource_id, remaining_demand, remaining_capacity = dead_resource
                    raise RuntimeError(
                        "Future resource demand already exceeds capacity "
                        f"for resource {resource_id + 1} "
                        f"(demand={remaining_demand}, capacity={remaining_capacity})."
                    )
                removed_jobs = self._rollback_resource_pressure_state(
                    state,
                    unscheduled_jobs,
                    resource_id=dead_resource[0],
                    window_start=0,
                    window_end=10**9,
                )
                if not removed_jobs:
                    raise RuntimeError("Could not build a feasible initial solution.")
                rollback_count += 1
                remaining_predecessors, ready_jobs = self._recompute_frontier(unscheduled_jobs, state.scheduled_jobs)
                resource_timelines = self._build_resource_timelines_from_state(state)
                continue

            ordered_ready = sorted(
                ready_jobs,
                key=lambda job_id: self._large_dynamic_job_priority_key(
                    state,
                    job_id,
                    remaining_resource_demand,
                    resource_timelines,
                ),
            )[: max(8, self.frontier_size)]

            best_move: Optional[CandidateMove] = None
            for job_id in ordered_ready:
                move = self._best_append_candidate_with_timelines(state, resource_timelines, job_id)
                if move is None:
                    continue
                if best_move is None or move.score < best_move.score:
                    best_move = move

            if best_move is None:
                if rollback_count >= self.max_rollbacks or not state.insertion_order:
                    blocked_diagnostics = self._blocked_job_diagnostics(ordered_ready[:5])
                    if blocked_diagnostics:
                        raise RuntimeError(
                            "Could not place ready job set; blocked ready jobs="
                            f"{blocked_diagnostics}."
                        )
                    blocked_label = ",".join(str(job_id + 1) for job_id in ordered_ready[:5]) or "none"
                    raise RuntimeError(f"Could not place ready job set; blocked ready jobs={blocked_label}.")

                removed_jobs = self._rollback_blocked_large_state(state, unscheduled_jobs, ordered_ready)
                if not removed_jobs:
                    raise RuntimeError("Could not build a feasible initial solution.")
                rollback_count += 1
                remaining_predecessors, ready_jobs = self._recompute_frontier(unscheduled_jobs, state.scheduled_jobs)
                resource_timelines = self._build_resource_timelines_from_state(state)
                continue

            self._apply_candidate(state, best_move)
            for resource_id, required_amount in self.instance.job_resource_requirements[best_move.job_id]:
                resource_timelines[resource_id].commit(best_move.start, best_move.finish, required_amount)

            unscheduled_jobs.remove(best_move.job_id)
            ready_jobs.remove(best_move.job_id)
            for successor_id in self.instance.successor_indices[best_move.job_id]:
                remaining_predecessors[successor_id] -= 1
                if remaining_predecessors[successor_id] == 0 and successor_id in unscheduled_jobs:
                    ready_jobs.add(successor_id)

        return self._finalize_large_state(state)

    def _finalize_large_state(self, state: ScheduleState) -> Dict:
        objective, tardiness, makespan, feasible, solution, _, _, _ = self.instance.decode_sequences(
            state.machine_sequences
        )
        if not feasible:
            raise RuntimeError("Constructed schedule failed final feasibility decoding.")

        return {
            "objective": objective,
            "tardiness": tardiness,
            "makespan": makespan,
            "solution": solution,
            "job_sequences_by_machine": state.machine_sequences,
            "feasible": True,
            "infeasibility_score": 0,
            "violation_summary": {
                "ineligible_jobs": 0,
                "precedence_shortfall": 0,
                "setup_shortfall": 0,
                "machine_overlap": 0,
                "resource_overload": 0,
                "violating_jobs": 0,
            },
        }

    def beam_search_large_instance(
        self,
        deadline: Optional[float],
        beam_width: int = 4,
        branch_limit: int = 3,
    ) -> Dict:
        initial_node = BeamNode(
            state=self._empty_state(),
            unscheduled_jobs=set(range(self.instance.n_jobs)),
            remaining_predecessors=[len(preds) for preds in self.instance.predecessor_indices],
            ready_jobs={
                job_id
                for job_id, count in enumerate(
                    [len(preds) for preds in self.instance.predecessor_indices]
                )
                if count == 0
            },
            resource_timelines=self.instance._build_resource_timelines(),
        )
        beam: List[BeamNode] = [initial_node]
        blocked_jobs_seen: List[int] = []

        while beam:
            if self._time_exceeded(deadline):
                raise RuntimeError("Timed out while building a feasible initial solution.")

            expanded_nodes: List[Tuple[Tuple[float, int, int, int], BeamNode]] = []
            progress_made = False

            for node in beam:
                if not node.unscheduled_jobs:
                    return self._finalize_large_state(node.state)

                remaining_resource_demand = self._remaining_resource_demand(node.unscheduled_jobs)
                dead_resource = self._find_aggregate_resource_impossibility(
                    remaining_resource_demand,
                    node.resource_timelines,
                )
                if dead_resource is not None:
                    continue

                if not node.ready_jobs:
                    continue

                critical_job_id = self._select_beam_critical_job(
                    node.state,
                    node.ready_jobs,
                    remaining_resource_demand,
                    node.resource_timelines,
                )
                if critical_job_id is None:
                    continue

                candidate_moves = self._append_candidates_with_timelines(
                    node.state,
                    node.resource_timelines,
                    critical_job_id,
                    branch_limit,
                )
                if not candidate_moves:
                    blocked_jobs_seen.append(critical_job_id)
                    continue

                progress_made = True
                for candidate_move in candidate_moves:
                    child = self._clone_beam_node(node)
                    self._apply_candidate(child.state, candidate_move)
                    for resource_id, required_amount in self.instance.job_resource_requirements[candidate_move.job_id]:
                        child.resource_timelines[resource_id].commit(
                            candidate_move.start,
                            candidate_move.finish,
                            required_amount,
                        )
                    child.unscheduled_jobs.remove(candidate_move.job_id)
                    child.ready_jobs.remove(candidate_move.job_id)
                    for successor_id in self.instance.successor_indices[candidate_move.job_id]:
                        child.remaining_predecessors[successor_id] -= 1
                        if (
                            child.remaining_predecessors[successor_id] == 0
                            and successor_id in child.unscheduled_jobs
                        ):
                            child.ready_jobs.add(successor_id)

                    if not child.unscheduled_jobs:
                        return self._finalize_large_state(child.state)

                    expanded_nodes.append(
                        (
                            self._score_beam_node(
                                child.state,
                                child.unscheduled_jobs,
                                child.resource_timelines,
                            ),
                            child,
                        )
                    )

            if not progress_made or not expanded_nodes:
                blocked_label = ",".join(
                    str(job_id + 1) for job_id in blocked_jobs_seen[-5:]
                ) or "none"
                raise RuntimeError(f"Could not place ready job set; blocked ready jobs={blocked_label}.")

            expanded_nodes.sort(key=lambda item: item[0])
            beam = [node for _, node in expanded_nodes[:beam_width]]

        raise RuntimeError("Could not build a feasible initial solution.")

    def large_instance_repair_lns(
        self,
        deadline: Optional[float],
        initial_sequences: Optional[List[List[int]]] = None,
    ) -> Dict:
        current_sequences = (
            self._clone_machine_sequences(initial_sequences)
            if initial_sequences is not None
            else self._construct_complete_machine_sequences()
        )
        current_eval = self._evaluate_complete_schedule(current_sequences)
        best_sequences = self._clone_machine_sequences(current_sequences)
        best_eval = current_eval
        non_improving_iterations = 0
        iteration_limit = max(200, self.max_rollbacks * 12)

        for _ in range(iteration_limit):
            if self._time_exceeded(deadline):
                break
            if best_eval.feasible and non_improving_iterations >= max(10, self.max_rollbacks // 2):
                break

            candidate_sequences = self._repair_overload_peak_neighborhood(current_sequences, current_eval)
            if candidate_sequences is None:
                candidate_sequences = self._repair_resource_peak_multi_shift(current_sequences, current_eval)
            if candidate_sequences is None:
                candidate_sequences = self._repair_resource_overload(current_sequences, current_eval)
            if candidate_sequences is None:
                destroyed_jobs = self._select_lns_destroy_jobs(current_sequences, current_eval)
                candidate_sequences = self._reinsert_destroyed_jobs(current_sequences, destroyed_jobs)
            candidate_eval = self._evaluate_complete_schedule(candidate_sequences)

            if self._is_better_evaluation(candidate_eval, current_eval):
                current_sequences = candidate_sequences
                current_eval = candidate_eval
                if self._is_better_evaluation(candidate_eval, best_eval):
                    best_sequences = self._clone_machine_sequences(candidate_sequences)
                    best_eval = candidate_eval
                non_improving_iterations = 0
            else:
                non_improving_iterations += 1
                if not best_eval.feasible and non_improving_iterations >= max(20, self.max_rollbacks):
                    current_sequences = self._clone_machine_sequences(best_sequences)
                    current_eval = best_eval
                    non_improving_iterations = 0

        return {
            "objective": best_eval.objective,
            "tardiness": best_eval.tardiness,
            "makespan": best_eval.makespan,
            "solution": best_eval.solution,
            "job_sequences_by_machine": best_sequences,
            "feasible": best_eval.feasible,
            "infeasibility_score": best_eval.infeasibility_score,
            "violation_summary": best_eval.violation_summary,
        }

    def repaired_initial_feasible_solution(self, deadline: Optional[float]) -> Dict:
        state = self._empty_state()
        unscheduled_jobs = set(range(self.instance.n_jobs))
        remaining_predecessors = [len(preds) for preds in self.instance.predecessor_indices]
        ready_jobs = {job_id for job_id, count in enumerate(remaining_predecessors) if count == 0}
        rollback_count = 0

        while unscheduled_jobs:
            if self._time_exceeded(deadline):
                raise RuntimeError("Timed out while building a feasible initial solution.")
            candidates = self._ranked_candidates(state, ready_jobs, rollback_count)
            if not candidates:
                if rollback_count >= self.max_rollbacks:
                    raise RuntimeError("Could not build a feasible initial solution.")

                removed_jobs = self._rollback_conflict_neighborhood(state, unscheduled_jobs, ready_jobs)
                if not removed_jobs:
                    raise RuntimeError("Could not build a feasible initial solution.")

                rollback_count += 1
                remaining_predecessors, ready_jobs = self._recompute_frontier(unscheduled_jobs, state.scheduled_jobs)
                continue

            candidate = candidates[0]
            self._apply_candidate(state, candidate)
            unscheduled_jobs.remove(candidate.job_id)
            ready_jobs.remove(candidate.job_id)
            for successor_id in self.instance.successor_indices[candidate.job_id]:
                remaining_predecessors[successor_id] -= 1
                if remaining_predecessors[successor_id] == 0 and successor_id in unscheduled_jobs:
                    ready_jobs.add(successor_id)

        tardiness = sum(
            max(0, state.finish_times[job_id] - self.instance.due[job_id])
            for job_id in range(self.instance.n_jobs)
        )
        makespan = max(state.finish_times, default=0)
        return {
            "objective": tardiness + makespan,
            "tardiness": tardiness,
            "makespan": makespan,
            "solution": self.instance.solution_from_schedule(state.start_times, state.machine_assignment),
            "job_sequences_by_machine": state.machine_sequences,
        }

    def randomized_append_solution(
        self,
        max_restarts: int = 200,
        candidate_pool_size: int = 5,
        ready_job_limit: int = 20,
        deadline: Optional[float] = None,
    ) -> Dict:
        for _ in range(max_restarts):
            if self._time_exceeded(deadline):
                raise RuntimeError("Timed out while building a feasible initial solution.")
            machine_sequences = [[] for _ in range(self.instance.n_machines + 1)]
            unscheduled_jobs = set(range(self.instance.n_jobs))
            scheduled_jobs: Set[int] = set()
            start_times = [0] * self.instance.n_jobs
            finish_times = [0] * self.instance.n_jobs
            machine_ready_time = [0] * (self.instance.n_machines + 1)
            previous_job_by_machine = [None] * (self.instance.n_machines + 1)

            while unscheduled_jobs:
                if self._time_exceeded(deadline):
                    raise RuntimeError("Timed out while building a feasible initial solution.")
                candidate_moves: List[CandidateMove] = []
                ready_jobs = [
                    job_id
                    for job_id in unscheduled_jobs
                    if all(pred_id in scheduled_jobs for pred_id in self.instance.predecessor_indices[job_id])
                ]
                ready_jobs.sort(key=lambda job_id: self._randomized_priority(job_id, finish_times, scheduled_jobs))
                if len(ready_jobs) > ready_job_limit:
                    ready_jobs = ready_jobs[:ready_job_limit]

                for job_id in ready_jobs:
                    predecessor_finish = max(
                        (finish_times[pred_id] for pred_id in self.instance.predecessor_indices[job_id]),
                        default=0,
                    )
                    eligible_machines = list(self.instance.eligible[job_id])
                    self.rng.shuffle(eligible_machines)
                    for machine_id in eligible_machines:
                        if previous_job_by_machine[machine_id] is None:
                            candidate_start = max(predecessor_finish, self.instance.initial_setup[job_id])
                        else:
                            candidate_start = max(
                                predecessor_finish,
                                machine_ready_time[machine_id]
                                + self.instance.setup[job_id][previous_job_by_machine[machine_id]],
                            )
                        candidate_finish = candidate_start + self.instance.processing[job_id]
                        if self.instance.has_resource_conflict(
                            job_id,
                            candidate_start,
                            candidate_finish,
                            scheduled_jobs,
                            start_times,
                            finish_times,
                        ):
                            continue
                        tardiness_increase = max(0, candidate_finish - self.instance.due[job_id])
                        score = (
                            1000 * tardiness_increase + candidate_finish + self.instance.resource_weights[job_id],
                            len(self.instance.eligible[job_id]),
                            job_id,
                        )
                        candidate_moves.append(
                            CandidateMove(score, job_id, machine_id, len(machine_sequences[machine_id]), candidate_start, candidate_finish)
                        )

                if not candidate_moves:
                    break

                candidate_moves.sort(key=lambda move: move.score)
                pool = candidate_moves[: min(candidate_pool_size, len(candidate_moves))]
                chosen = self.rng.choice(pool)
                machine_sequences[chosen.machine_id].append(chosen.job_id)
                start_times[chosen.job_id] = chosen.start
                finish_times[chosen.job_id] = chosen.finish
                machine_ready_time[chosen.machine_id] = chosen.finish
                previous_job_by_machine[chosen.machine_id] = chosen.job_id
                scheduled_jobs.add(chosen.job_id)
                unscheduled_jobs.remove(chosen.job_id)

            if unscheduled_jobs:
                continue

            objective, tardiness, makespan, feasible, solution, _, _, _ = self.instance.decode_sequences(machine_sequences)
            if feasible:
                return {
                    "objective": objective,
                    "tardiness": tardiness,
                    "makespan": makespan,
                    "solution": solution,
                    "job_sequences_by_machine": machine_sequences,
                }

        raise RuntimeError("Could not build a feasible initial solution.")

    def _randomized_priority(
        self,
        job_id: int,
        finish_times: List[int],
        scheduled_jobs: Set[int],
    ) -> Tuple[int, int, int, int, float]:
        predecessor_finish = max(
            (finish_times[pred_id] for pred_id in self.instance.predecessor_indices[job_id] if pred_id in scheduled_jobs),
            default=0,
        )
        earliest_finish = predecessor_finish + self.instance.processing[job_id]
        dynamic_slack = self.instance.due[job_id] - earliest_finish
        return (
            len(self.instance.eligible[job_id]),
            dynamic_slack,
            -self.instance.resource_weights[job_id],
            -len(self.instance.successor_indices[job_id]),
            self.rng.random(),
        )

    def _time_exceeded(self, deadline: Optional[float]) -> bool:
        return deadline is not None and time.time() >= deadline

    def _empty_state(self) -> ScheduleState:
        return ScheduleState(
            machine_sequences=[[] for _ in range(self.instance.n_machines + 1)],
            machine_assignment=[0] * self.instance.n_jobs,
            start_times=[-1] * self.instance.n_jobs,
            finish_times=[-1] * self.instance.n_jobs,
            machine_ready_time=[0] * (self.instance.n_machines + 1),
            previous_job_by_machine=[None] * (self.instance.n_machines + 1),
            resource_usage=[[] for _ in range(self.instance.n_resources)],
            scheduled_jobs=set(),
            insertion_order=[],
        )

    def _clone_state(self, state: ScheduleState) -> ScheduleState:
        return ScheduleState(
            machine_sequences=[list(sequence) for sequence in state.machine_sequences],
            machine_assignment=list(state.machine_assignment),
            start_times=list(state.start_times),
            finish_times=list(state.finish_times),
            machine_ready_time=list(state.machine_ready_time),
            previous_job_by_machine=list(state.previous_job_by_machine),
            resource_usage=[list(intervals) for intervals in state.resource_usage],
            scheduled_jobs=set(state.scheduled_jobs),
            insertion_order=list(state.insertion_order),
        )

    def _clone_resource_timelines(self, resource_timelines: List[ResourceTimeline]) -> List[ResourceTimeline]:
        cloned: List[ResourceTimeline] = []
        for timeline in resource_timelines:
            new_timeline = ResourceTimeline([])
            new_timeline.segments = [list(segment) for segment in timeline.segments]
            cloned.append(new_timeline)
        return cloned

    def _clone_beam_node(self, node: BeamNode) -> BeamNode:
        return BeamNode(
            state=self._clone_state(node.state),
            unscheduled_jobs=set(node.unscheduled_jobs),
            remaining_predecessors=list(node.remaining_predecessors),
            ready_jobs=set(node.ready_jobs),
            resource_timelines=self._clone_resource_timelines(node.resource_timelines),
        )

    def _ranked_candidates(
        self, state: ScheduleState, ready_jobs: Set[int], diversification_level: int
    ) -> List[CandidateMove]:
        if not ready_jobs:
            return []

        ordered_jobs = sorted(ready_jobs, key=lambda job_id: self._dynamic_job_priority_key(state, job_id))
        restricted_jobs = ordered_jobs[: self.frontier_size]
        ranked = self._best_candidate_over_jobs(state, restricted_jobs)
        if ranked:
            return self._diversify_candidates(ranked, diversification_level)
        if len(restricted_jobs) == len(ordered_jobs):
            return []
        ranked = self._best_candidate_over_jobs(state, ordered_jobs)
        return self._diversify_candidates(ranked, diversification_level)

    def _best_append_candidate_with_timelines(
        self,
        state: ScheduleState,
        resource_timelines,
        job_id: int,
    ) -> Optional[CandidateMove]:
        candidates = self._append_candidates_with_timelines(
            state,
            resource_timelines,
            job_id,
            limit=1,
        )
        return candidates[0] if candidates else None

    def _append_candidates_with_timelines(
        self,
        state: ScheduleState,
        resource_timelines,
        job_id: int,
        limit: int,
    ) -> List[CandidateMove]:
        predecessor_finish = self._predecessor_finish(state, job_id)
        candidates: List[CandidateMove] = []

        for machine_id in self.instance.eligible[job_id]:
            feasible_start = max(predecessor_finish, self._earliest_machine_ready(state, job_id, machine_id))
            while True:
                updated_start = feasible_start
                for resource_id, required_amount in self.instance.job_resource_requirements[job_id]:
                    candidate_start = resource_timelines[resource_id].earliest_feasible_start(
                        updated_start,
                        self.instance.processing[job_id],
                        required_amount,
                    )
                    if candidate_start >= 10**9:
                        updated_start = 10**9
                        break
                    updated_start = max(updated_start, candidate_start)
                if updated_start >= 10**9:
                    feasible_start = 10**9
                    break
                if updated_start == feasible_start:
                    break
                feasible_start = updated_start

            if feasible_start >= 10**9:
                continue

            finish = feasible_start + self.instance.processing[job_id]
            tardiness_increase = max(0, finish - self.instance.due[job_id])
            regret = self._job_machine_regret(state, job_id, resource_timelines)
            score = (
                1000 * tardiness_increase + finish + self.instance.resource_weights[job_id],
                len(self.instance.eligible[job_id]),
                -regret,
                -len(self.instance.successor_indices[job_id]),
                job_id,
                machine_id,
            )
            candidates.append(
                CandidateMove(
                    score,
                    job_id,
                    machine_id,
                    len(state.machine_sequences[machine_id]),
                    feasible_start,
                    finish,
                )
            )
        candidates.sort(key=lambda move: move.score)
        return candidates[:limit]

    def _construct_complete_machine_sequences(self) -> List[List[int]]:
        machine_sequences = [[] for _ in range(self.instance.n_machines + 1)]
        machine_ready_time = [0] * (self.instance.n_machines + 1)
        machine_last_job = [None] * (self.instance.n_machines + 1)
        remaining_predecessors = [len(preds) for preds in self.instance.predecessor_indices]
        ready_heap: List[Tuple[Tuple[int, int, int, int, int], int]] = []

        for job_id, count in enumerate(remaining_predecessors):
            if count == 0:
                heapq.heappush(ready_heap, (self.instance.job_priority_key(job_id), job_id))

        scheduled_count = 0
        while ready_heap:
            _, job_id = heapq.heappop(ready_heap)
            best_machine = self.instance.eligible[job_id][0]
            best_score: Optional[Tuple[int, int, int]] = None
            for machine_id in self.instance.eligible[job_id]:
                if machine_last_job[machine_id] is None:
                    start = self.instance.initial_setup[job_id]
                else:
                    start = machine_ready_time[machine_id] + self.instance.setup[job_id][machine_last_job[machine_id]]
                finish = start + self.instance.processing[job_id]
                score = (finish, len(machine_sequences[machine_id]), machine_id)
                if best_score is None or score < best_score:
                    best_score = score
                    best_machine = machine_id

            machine_sequences[best_machine].append(job_id)
            previous_job = machine_last_job[best_machine]
            if previous_job is None:
                machine_ready_time[best_machine] = self.instance.initial_setup[job_id] + self.instance.processing[job_id]
            else:
                machine_ready_time[best_machine] += self.instance.setup[job_id][previous_job] + self.instance.processing[job_id]
            machine_last_job[best_machine] = job_id
            scheduled_count += 1

            for successor_id in self.instance.successor_indices[job_id]:
                remaining_predecessors[successor_id] -= 1
                if remaining_predecessors[successor_id] == 0:
                    heapq.heappush(ready_heap, (self.instance.job_priority_key(successor_id), successor_id))

        if scheduled_count != self.instance.n_jobs:
            raise RuntimeError("Could not construct a complete machine ordering.")
        return machine_sequences

    def _relaxed_decode_sequences(
        self,
        machine_sequences: List[List[int]],
    ) -> Tuple[bool, List[int], List[int], List[int]]:
        machine_assignment = [0] * self.instance.n_jobs
        seen_jobs: Set[int] = set()
        for machine_id in range(1, self.instance.n_machines + 1):
            for job_id in machine_sequences[machine_id]:
                if job_id in seen_jobs:
                    return False, [], [], []
                seen_jobs.add(job_id)
                machine_assignment[job_id] = machine_id

        if len(seen_jobs) != self.instance.n_jobs:
            return False, [], [], []

        start_times = [0] * self.instance.n_jobs
        finish_times = [0] * self.instance.n_jobs
        scheduled_jobs: Set[int] = set()
        next_job_pos = [0] * (self.instance.n_machines + 1)
        machine_ready_time = [0] * (self.instance.n_machines + 1)
        previous_job_by_machine = [None] * (self.instance.n_machines + 1)

        while len(scheduled_jobs) < self.instance.n_jobs:
            progress = False
            for machine_id in range(1, self.instance.n_machines + 1):
                job_pos = next_job_pos[machine_id]
                sequence = machine_sequences[machine_id]
                if job_pos >= len(sequence):
                    continue

                job_id = sequence[job_pos]
                predecessors = self.instance.predecessor_indices[job_id]
                if any(pred_id not in scheduled_jobs for pred_id in predecessors):
                    continue

                previous_job_id = previous_job_by_machine[machine_id]
                if previous_job_id is None:
                    machine_time = max(machine_ready_time[machine_id], self.instance.initial_setup[job_id])
                else:
                    machine_time = (
                        machine_ready_time[machine_id]
                        + self.instance.setup[job_id][previous_job_id]
                    )

                predecessor_finish = max((finish_times[pred_id] for pred_id in predecessors), default=0)
                candidate_start = max(machine_time, predecessor_finish)
                candidate_finish = candidate_start + self.instance.processing[job_id]

                start_times[job_id] = candidate_start
                finish_times[job_id] = candidate_finish
                machine_ready_time[machine_id] = candidate_finish
                previous_job_by_machine[machine_id] = job_id
                next_job_pos[machine_id] += 1
                scheduled_jobs.add(job_id)
                progress = True

            if not progress:
                return False, [], [], []

        return True, start_times, finish_times, machine_assignment

    def _evaluate_complete_schedule(self, machine_sequences: List[List[int]]) -> ScheduleEvaluation:
        decoded, start_times, finish_times, machine_assignment = self._relaxed_decode_sequences(machine_sequences)
        if not decoded:
            huge = 10**12
            return ScheduleEvaluation(
                feasible=False,
                infeasibility_score=huge,
                objective=huge,
                tardiness=huge,
                makespan=huge,
                solution={"Jobs": []},
                start_times=[],
                finish_times=[],
                machine_assignment=[],
                violating_jobs=set(),
                violation_summary={"decode_failure": huge},
                overload_events=[],
            )

        violating_jobs: Set[int] = set()
        ineligible_jobs = 0
        precedence_shortfall = 0
        setup_shortfall = 0
        machine_overlap = 0

        for job_id in range(self.instance.n_jobs):
            machine_id = machine_assignment[job_id]
            if machine_id not in self.instance.eligible[job_id]:
                ineligible_jobs += 1
                violating_jobs.add(job_id)

            for pred_id in self.instance.predecessor_indices[job_id]:
                shortfall = max(0, finish_times[pred_id] - start_times[job_id])
                if shortfall > 0:
                    precedence_shortfall += shortfall
                    violating_jobs.add(job_id)
                    violating_jobs.add(pred_id)

        for machine_id in range(1, self.instance.n_machines + 1):
            sequence = machine_sequences[machine_id]
            if not sequence:
                continue

            first_job = sequence[0]
            initial_shortfall = max(0, self.instance.initial_setup[first_job] - start_times[first_job])
            if initial_shortfall > 0:
                setup_shortfall += initial_shortfall
                violating_jobs.add(first_job)

            for pos in range(1, len(sequence)):
                prev_job = sequence[pos - 1]
                job_id = sequence[pos]
                prev_finish = finish_times[prev_job]
                overlap = max(0, prev_finish - start_times[job_id])
                if overlap > 0:
                    machine_overlap += overlap
                    violating_jobs.add(prev_job)
                    violating_jobs.add(job_id)
                required_setup = self.instance.setup[job_id][prev_job]
                actual_gap = start_times[job_id] - prev_finish
                shortfall = max(0, required_setup - actual_gap)
                if shortfall > 0:
                    setup_shortfall += shortfall
                    violating_jobs.add(prev_job)
                    violating_jobs.add(job_id)

        resource_overload = 0
        overload_events: List[Dict] = []
        for resource_id in range(self.instance.n_resources):
            events: List[Tuple[int, int, int, int]] = []
            for period_start, _, capacity in self.instance.resource_periods[resource_id]:
                events.append((period_start, 1, capacity, -1))
            if self.instance.resource_periods[resource_id]:
                events.append((self.instance.resource_periods[resource_id][-1][1], 1, 0, -1))

            for job_id in range(self.instance.n_jobs):
                for required_resource_id, capacity in self.instance.job_resource_requirements[job_id]:
                    if required_resource_id != resource_id:
                        continue
                    events.append((start_times[job_id], 2, capacity, job_id))
                    events.append((finish_times[job_id], 0, capacity, job_id))

            events.sort()
            current_capacity = 0
            active_jobs: Dict[int, int] = {}
            for time_value, event_type, capacity, job_id in events:
                if event_type == 0:
                    active_jobs.pop(job_id, None)
                    current_capacity += capacity
                elif event_type == 1:
                    current_capacity = capacity
                else:
                    if current_capacity < capacity:
                        missing = capacity - current_capacity
                        resource_overload += missing
                        violating_jobs.add(job_id)
                        overload_events.append(
                            {
                                "resource_id": resource_id,
                                "time": time_value,
                                "missing": missing,
                                "job_id": job_id,
                                "active_jobs": sorted(active_jobs.keys()) + [job_id],
                            }
                        )
                    current_capacity -= capacity
                    active_jobs[job_id] = capacity

        tardiness = sum(max(0, finish_times[job_id] - self.instance.due[job_id]) for job_id in range(self.instance.n_jobs))
        makespan = max(finish_times, default=0)
        infeasibility_score = (
            10**9 * ineligible_jobs
            + 10**6 * len(violating_jobs)
            + 10**4 * precedence_shortfall
            + 10**3 * (setup_shortfall + machine_overlap + resource_overload)
        )
        feasible = infeasibility_score == 0
        objective = tardiness + makespan
        if not feasible:
            objective += infeasibility_score

        violation_summary = {
            "ineligible_jobs": ineligible_jobs,
            "precedence_shortfall": precedence_shortfall,
            "setup_shortfall": setup_shortfall,
            "machine_overlap": machine_overlap,
            "resource_overload": resource_overload,
            "violating_jobs": len(violating_jobs),
        }

        return ScheduleEvaluation(
            feasible=feasible,
            infeasibility_score=infeasibility_score,
            objective=objective,
            tardiness=tardiness,
            makespan=makespan,
            solution=self.instance.solution_from_schedule(start_times, machine_assignment),
            start_times=start_times,
            finish_times=finish_times,
            machine_assignment=machine_assignment,
            violating_jobs=violating_jobs,
            violation_summary=violation_summary,
            overload_events=sorted(
                overload_events,
                key=lambda event: (-event["missing"], event["time"], event["resource_id"], event["job_id"]),
            ),
        )

    def _select_lns_destroy_jobs(
        self,
        machine_sequences: List[List[int]],
        evaluation: ScheduleEvaluation,
    ) -> List[int]:
        if evaluation.overload_events:
            hottest_event = evaluation.overload_events[0]
            candidate_jobs = list(hottest_event["active_jobs"])
        else:
            candidate_jobs = list(evaluation.violating_jobs)
        if not candidate_jobs:
            candidate_jobs = list(range(self.instance.n_jobs))
            self.rng.shuffle(candidate_jobs)

        seeds = candidate_jobs[: max(2, self.rollback_size)]
        destroy_set: List[int] = []
        seen: Set[int] = set()
        target_size = max(self.rollback_size * 2, 12)

        for job_id in seeds:
            if job_id not in seen:
                destroy_set.append(job_id)
                seen.add(job_id)

            machine_id = evaluation.machine_assignment[job_id]
            if machine_id != 0:
                sequence = machine_sequences[machine_id]
                pos = sequence.index(job_id)
                for neighbor_pos in range(max(0, pos - 1), min(len(sequence), pos + 2)):
                    neighbor_job = sequence[neighbor_pos]
                    if neighbor_job not in seen:
                        destroy_set.append(neighbor_job)
                        seen.add(neighbor_job)

            for neighbor_job in self.instance.predecessor_indices[job_id] + self.instance.successor_indices[job_id]:
                if neighbor_job not in seen:
                    destroy_set.append(neighbor_job)
                    seen.add(neighbor_job)
                if len(destroy_set) >= target_size:
                    return destroy_set

            if len(destroy_set) >= target_size:
                return destroy_set

        return destroy_set

    def _repair_resource_peak_multi_shift(
        self,
        machine_sequences: List[List[int]],
        evaluation: ScheduleEvaluation,
    ) -> Optional[List[List[int]]]:
        if not evaluation.overload_events:
            return None

        hottest_event = evaluation.overload_events[0]
        selected_jobs = self._select_peak_shift_jobs(hottest_event, evaluation)
        if len(selected_jobs) < 2:
            return None

        base_sequences = self._clone_machine_sequences(machine_sequences)
        removed_jobs: List[int] = []
        for machine_id in range(1, self.instance.n_machines + 1):
            kept_sequence = []
            for job_id in base_sequences[machine_id]:
                if job_id in selected_jobs:
                    removed_jobs.append(job_id)
                else:
                    kept_sequence.append(job_id)
            base_sequences[machine_id] = kept_sequence

        working_sequences = base_sequences
        reinsertion_order = sorted(
            removed_jobs,
            key=lambda job_id: (
                len(self.instance.eligible[job_id]),
                -sum(cap for _, cap in self.instance.job_resource_requirements[job_id]),
                -len(self.instance.successor_indices[job_id]),
                job_id,
            ),
        )

        for job_id in reinsertion_order:
            next_sequences = self._heuristic_reinsert_job(
                working_sequences,
                evaluation,
                job_id,
                hottest_event["time"],
            )
            if next_sequences is None:
                return None
            working_sequences = next_sequences

        partial_eval = self._evaluate_complete_schedule(working_sequences)
        if self._is_better_evaluation(partial_eval, evaluation):
            return working_sequences
        return None

    def _repair_overload_peak_neighborhood(
        self,
        machine_sequences: List[List[int]],
        evaluation: ScheduleEvaluation,
    ) -> Optional[List[List[int]]]:
        if not evaluation.overload_events:
            return None

        hottest_event = evaluation.overload_events[0]
        destroyed_jobs = self._select_overload_peak_neighborhood_jobs(
            machine_sequences,
            evaluation,
            hottest_event,
        )
        if len(destroyed_jobs) < max(6, self.rollback_size):
            return None

        candidate_sequences = self._reinsert_destroyed_jobs(machine_sequences, destroyed_jobs)
        candidate_eval = self._evaluate_complete_schedule(candidate_sequences)
        if self._is_better_evaluation(candidate_eval, evaluation):
            return candidate_sequences
        return None

    def _select_overload_peak_neighborhood_jobs(
        self,
        machine_sequences: List[List[int]],
        evaluation: ScheduleEvaluation,
        hottest_event: Dict,
    ) -> List[int]:
        resource_id = hottest_event["resource_id"]
        hotspot_time = hottest_event["time"]
        active_jobs = list(hottest_event["active_jobs"])
        active_jobs.sort(
            key=lambda job_id: (
                -sum(
                    cap
                    for req_resource_id, cap in self.instance.job_resource_requirements[job_id]
                    if req_resource_id == resource_id
                ),
                len(self.instance.eligible[job_id]),
                -len(self.instance.successor_indices[job_id]),
                job_id,
            )
        )

        target_size = min(
            self.instance.n_jobs,
            max(14, min(40, self.rollback_size * 4)),
        )
        destroy_set: List[int] = []
        seen: Set[int] = set()

        def add_job(job_id: int) -> None:
            if job_id not in seen:
                seen.add(job_id)
                destroy_set.append(job_id)

        for job_id in active_jobs:
            add_job(job_id)
            machine_id = evaluation.machine_assignment[job_id]
            if machine_id != 0:
                sequence = machine_sequences[machine_id]
                pos = sequence.index(job_id)
                for neighbor_pos in range(max(0, pos - 2), min(len(sequence), pos + 3)):
                    add_job(sequence[neighbor_pos])

            for neighbor_job in self.instance.predecessor_indices[job_id]:
                add_job(neighbor_job)
            for neighbor_job in self.instance.successor_indices[job_id]:
                add_job(neighbor_job)
            if len(destroy_set) >= target_size:
                return destroy_set[:target_size]

        for machine_id in range(1, self.instance.n_machines + 1):
            sequence = machine_sequences[machine_id]
            for idx, scheduled_job in enumerate(sequence):
                job_start = evaluation.start_times[scheduled_job]
                job_finish = evaluation.finish_times[scheduled_job]
                if job_start <= hotspot_time <= job_finish:
                    for neighbor_pos in range(max(0, idx - 2), min(len(sequence), idx + 3)):
                        add_job(sequence[neighbor_pos])
                    if len(destroy_set) >= target_size:
                        return destroy_set[:target_size]

        for job_id in evaluation.violating_jobs:
            add_job(job_id)
            if len(destroy_set) >= target_size:
                break

        return destroy_set[:target_size]

    def _select_peak_shift_jobs(
        self,
        hottest_event: Dict,
        evaluation: ScheduleEvaluation,
    ) -> Set[int]:
        resource_id = hottest_event["resource_id"]
        jobs = list(hottest_event["active_jobs"])
        jobs.sort(
            key=lambda job_id: (
                -sum(cap for req_resource_id, cap in self.instance.job_resource_requirements[job_id] if req_resource_id == resource_id),
                len(self.instance.eligible[job_id]),
                -evaluation.start_times[job_id],
                -len(self.instance.successor_indices[job_id]),
                job_id,
            )
        )
        target_size = min(len(jobs), max(3, self.rollback_size))
        return set(jobs[:target_size])

    def _repair_resource_overload(
        self,
        machine_sequences: List[List[int]],
        evaluation: ScheduleEvaluation,
    ) -> Optional[List[List[int]]]:
        if not evaluation.overload_events:
            return None

        hottest_event = evaluation.overload_events[0]
        candidate_jobs = list(hottest_event["active_jobs"])
        candidate_jobs.sort(
            key=lambda job_id: (
                -sum(cap for resource_id, cap in self.instance.job_resource_requirements[job_id] if resource_id == hottest_event["resource_id"]),
                len(self.instance.eligible[job_id]),
                -len(self.instance.successor_indices[job_id]),
                -evaluation.start_times[job_id],
                job_id,
            )
        )

        for culprit_job in candidate_jobs[: min(5, len(candidate_jobs))]:
            repaired = self._move_job_to_reduce_overload(machine_sequences, evaluation, culprit_job)
            if repaired is not None:
                return repaired
        return None

    def _move_job_to_reduce_overload(
        self,
        machine_sequences: List[List[int]],
        evaluation: ScheduleEvaluation,
        culprit_job: int,
    ) -> Optional[List[List[int]]]:
        current_machine = evaluation.machine_assignment[culprit_job]
        best_sequences: Optional[List[List[int]]] = None
        best_eval: Optional[ScheduleEvaluation] = None

        for machine_id in self.instance.eligible[culprit_job]:
            candidate_sequences = self._clone_machine_sequences(machine_sequences)
            for seq_machine_id in range(1, self.instance.n_machines + 1):
                candidate_sequences[seq_machine_id] = [
                    job_id for job_id in candidate_sequences[seq_machine_id] if job_id != culprit_job
                ]

            insertion_positions = [len(candidate_sequences[machine_id])]
            if machine_id == current_machine:
                original_sequence = machine_sequences[machine_id]
                original_pos = original_sequence.index(culprit_job)
                insertion_positions.extend(
                    pos for pos in range(original_pos + 1, len(candidate_sequences[machine_id]) + 1)
                )

            for insert_pos in sorted(set(insertion_positions)):
                trial_sequences = self._clone_machine_sequences(candidate_sequences)
                trial_sequences[machine_id].insert(insert_pos, culprit_job)
                trial_eval = self._evaluate_complete_schedule(trial_sequences)
                if best_eval is None or self._is_better_evaluation(trial_eval, best_eval):
                    best_eval = trial_eval
                    best_sequences = trial_sequences

        if best_eval is not None and self._is_better_evaluation(best_eval, evaluation):
            return best_sequences
        return None

    def _heuristic_reinsert_job(
        self,
        machine_sequences: List[List[int]],
        evaluation: ScheduleEvaluation,
        job_id: int,
        hotspot_time: int,
    ) -> Optional[List[List[int]]]:
        best_sequences: Optional[List[List[int]]] = None
        best_score: Optional[Tuple[int, int, int, int]] = None

        for machine_id in self.instance.eligible[job_id]:
            sequence = machine_sequences[machine_id]
            candidate_positions = self._candidate_reinsertion_positions(
                machine_id,
                sequence,
                evaluation,
                hotspot_time,
            )
            for insert_pos in candidate_positions:
                trial_sequences = self._clone_machine_sequences(machine_sequences)
                trial_sequences[machine_id].insert(insert_pos, job_id)
                anchor_time = self._estimate_reinsertion_anchor(
                    trial_sequences,
                    machine_id,
                    insert_pos,
                    job_id,
                    evaluation,
                )
                score = (
                    max(0, hotspot_time - anchor_time),
                    len(trial_sequences[machine_id]),
                    self.instance.machine_job_counts[machine_id],
                    machine_id,
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_sequences = trial_sequences

        return best_sequences

    def _candidate_reinsertion_positions(
        self,
        machine_id: int,
        sequence: List[int],
        evaluation: ScheduleEvaluation,
        hotspot_time: int,
    ) -> List[int]:
        if not sequence:
            return [0]

        positions = {len(sequence)}
        for idx, scheduled_job in enumerate(sequence):
            if evaluation.start_times[scheduled_job] >= hotspot_time:
                positions.add(idx)
                positions.add(min(idx + 1, len(sequence)))

        if len(positions) == 1:
            positions.add(max(0, len(sequence) - 1))

        return sorted(positions)

    def _estimate_reinsertion_anchor(
        self,
        machine_sequences: List[List[int]],
        machine_id: int,
        insert_pos: int,
        job_id: int,
        evaluation: ScheduleEvaluation,
    ) -> int:
        sequence = machine_sequences[machine_id]
        if insert_pos <= 0 or len(sequence) == 1:
            return self.instance.initial_setup[job_id]

        prev_job = sequence[insert_pos - 1]
        if prev_job < len(evaluation.finish_times):
            return evaluation.finish_times[prev_job] + self.instance.setup[job_id][prev_job]

        return self.instance.initial_setup[job_id]

    def _reinsert_destroyed_jobs(
        self,
        base_sequences: List[List[int]],
        destroyed_jobs: List[int],
    ) -> List[List[int]]:
        candidate_sequences = self._clone_machine_sequences(base_sequences)
        destroyed_set = set(destroyed_jobs)
        removed_order: List[int] = []

        for machine_id in range(1, self.instance.n_machines + 1):
            remaining_sequence = []
            for job_id in candidate_sequences[machine_id]:
                if job_id in destroyed_set:
                    removed_order.append(job_id)
                else:
                    remaining_sequence.append(job_id)
            candidate_sequences[machine_id] = remaining_sequence

        remaining_predecessors = {
            job_id: sum(1 for pred_id in self.instance.predecessor_indices[job_id] if pred_id in destroyed_set)
            for job_id in destroyed_set
        }
        ready_jobs = [
            job_id for job_id in removed_order
            if remaining_predecessors.get(job_id, 0) == 0
        ]
        placed_jobs: Set[int] = set()

        while ready_jobs:
            ready_jobs.sort(key=lambda job_id: self.instance.job_priority_key(job_id))
            job_id = ready_jobs.pop(0)
            if job_id in placed_jobs:
                continue

            best_machine = self.instance.eligible[job_id][0]
            best_score: Optional[Tuple[int, int, int, float]] = None
            for machine_id in self.instance.eligible[job_id]:
                previous_job = candidate_sequences[machine_id][-1] if candidate_sequences[machine_id] else None
                if previous_job is None:
                    anchor = self.instance.initial_setup[job_id]
                else:
                    anchor = self.instance.processing[previous_job] + self.instance.setup[job_id][previous_job]
                score = (
                    anchor,
                    len(candidate_sequences[machine_id]),
                    self.instance.machine_job_counts[machine_id],
                    self.rng.random(),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_machine = machine_id

            candidate_sequences[best_machine].append(job_id)
            placed_jobs.add(job_id)

            for successor_id in self.instance.successor_indices[job_id]:
                if successor_id in destroyed_set:
                    remaining_predecessors[successor_id] -= 1
                    if remaining_predecessors[successor_id] == 0:
                        ready_jobs.append(successor_id)

        for job_id in removed_order:
            if job_id not in placed_jobs:
                candidate_sequences[self.instance.eligible[job_id][0]].append(job_id)

        return candidate_sequences

    def _is_better_evaluation(self, candidate: ScheduleEvaluation, incumbent: ScheduleEvaluation) -> bool:
        return (
            candidate.infeasibility_score,
            candidate.objective,
            candidate.tardiness,
            candidate.makespan,
        ) < (
            incumbent.infeasibility_score,
            incumbent.objective,
            incumbent.tardiness,
            incumbent.makespan,
        )

    def _clone_machine_sequences(self, machine_sequences: List[List[int]]) -> List[List[int]]:
        return [list(sequence) for sequence in machine_sequences]

    def _ranked_large_append_candidates(
        self,
        state: ScheduleState,
        ready_jobs: Set[int],
    ) -> Tuple[List[CandidateMove], List[int]]:
        if not ready_jobs:
            return [], []

        ordered_jobs = sorted(ready_jobs, key=lambda job_id: self._dynamic_job_priority_key(state, job_id))
        restricted_jobs = ordered_jobs[: max(8, min(self.frontier_size, len(ordered_jobs)))]
        candidates: List[CandidateMove] = []
        blocked_jobs: List[int] = []

        for job_id in restricted_jobs:
            best_move = self._best_append_candidate(state, job_id)
            if best_move is None:
                blocked_jobs.append(job_id)
                continue
            candidates.append(best_move)

        candidates.sort(key=lambda candidate: candidate.score)
        return candidates[: max(4, self.frontier_size // 2)], blocked_jobs or restricted_jobs[:3]

    def _best_append_candidate(self, state: ScheduleState, job_id: int) -> Optional[CandidateMove]:
        predecessor_finish = self._predecessor_finish(state, job_id)
        best_move: Optional[CandidateMove] = None

        for machine_id in self.instance.eligible[job_id]:
            earliest_start = max(predecessor_finish, self._earliest_machine_ready(state, job_id, machine_id))
            feasible_start = self.instance.earliest_resource_feasible_start(
                job_id,
                earliest_start,
                state.scheduled_jobs,
                state.start_times,
                state.finish_times,
            )
            if feasible_start >= 10**9:
                continue

            finish = feasible_start + self.instance.processing[job_id]
            tardiness_increase = max(0, finish - self.instance.due[job_id])
            score = (
                1000 * tardiness_increase + finish + self.instance.resource_weights[job_id],
                len(self.instance.eligible[job_id]),
                -len(self.instance.successor_indices[job_id]),
                job_id,
                machine_id,
            )
            move = CandidateMove(
                score,
                job_id,
                machine_id,
                len(state.machine_sequences[machine_id]),
                feasible_start,
                finish,
            )
            if best_move is None or move.score < best_move.score:
                best_move = move

        return best_move

    def _best_candidate_over_jobs(
        self, state: ScheduleState, candidate_jobs: List[int]
    ) -> List[CandidateMove]:
        candidates: List[CandidateMove] = []
        for job_id in candidate_jobs:
            predecessor_finish = self._predecessor_finish(state, job_id)
            for machine_id in self.instance.eligible[job_id]:
                for insert_pos, start, finish, slack_after in self._machine_insertions(
                    state, job_id, machine_id, predecessor_finish
                ):
                    feasible, resource_penalty = self._resource_feasible_with_penalty(
                        state.resource_usage, job_id, start, finish
                    )
                    if not feasible:
                        continue

                    tardiness_increase = max(0, finish - self.instance.due[job_id])
                    setup_anchor = predecessor_finish
                    if insert_pos > 0:
                        prev_job_id = state.machine_sequences[machine_id][insert_pos - 1]
                        setup_anchor = max(setup_anchor, state.finish_times[prev_job_id])
                    setup_cost = max(0, start - setup_anchor)
                    machine_penalty = self._machine_flexibility_penalty(job_id, machine_id)
                    dynamic_slack = self.instance.due[job_id] - finish
                    score = (
                        1000 * tardiness_increase
                        + 10 * setup_cost
                        + finish
                        + resource_penalty
                        + machine_penalty,
                        -dynamic_slack,
                        slack_after,
                        finish,
                        machine_penalty,
                        job_id,
                    )
                    candidates.append(CandidateMove(score, job_id, machine_id, insert_pos, start, finish))
        candidates.sort(key=lambda candidate: candidate.score)
        return candidates[: max(10, self.frontier_size)]

    def _diversify_candidates(
        self, candidates: List[CandidateMove], diversification_level: int
    ) -> List[CandidateMove]:
        if not candidates:
            return []
        window = min(8, len(candidates))
        prefix = candidates[:window]
        self.rng.shuffle(prefix)
        remainder = candidates[window:]
        offset = diversification_level % len(prefix)
        return prefix[offset:] + prefix[:offset] + remainder

    def _dynamic_job_priority_key(self, state: ScheduleState, job_id: int) -> Tuple[int, int, int, int, int, int]:
        predecessor_finish = self._predecessor_finish(state, job_id)
        earliest_machine_start = min(
            self._earliest_machine_ready(state, job_id, machine_id)
            for machine_id in self.instance.eligible[job_id]
        )
        earliest_finish = max(predecessor_finish, earliest_machine_start) + self.instance.processing[job_id]
        dynamic_slack = self.instance.due[job_id] - earliest_finish
        scarce_resource_penalty = self.instance.resource_weights[job_id] * max(
            1, len(self.instance.job_resource_requirements[job_id])
        )
        return (
            len(self.instance.eligible[job_id]),
            dynamic_slack,
            -scarce_resource_penalty,
            -len(self.instance.successor_indices[job_id]),
            -self.instance.processing[job_id],
            job_id,
        )

    def _large_dynamic_job_priority_key(
        self,
        state: ScheduleState,
        job_id: int,
        remaining_resource_demand: List[int],
        resource_timelines,
    ) -> Tuple[float, int, int, int, int, int, int]:
        base_key = self._dynamic_job_priority_key(state, job_id)
        pressure = self._job_resource_pressure(job_id, remaining_resource_demand, resource_timelines)
        predecessor_finish = self._predecessor_finish(state, job_id)
        earliest_machine_start = min(
            self._earliest_machine_ready(state, job_id, machine_id)
            for machine_id in self.instance.eligible[job_id]
        )
        earliest_finish = max(predecessor_finish, earliest_machine_start) + self.instance.processing[job_id]
        regret = self._job_machine_regret(state, job_id, resource_timelines)
        return (
            -pressure,
            len(self.instance.eligible[job_id]),
            -regret,
            base_key[1],
            -len(self.instance.successor_indices[job_id]),
            earliest_finish,
            job_id,
        )

    def _job_resource_pressure(
        self,
        job_id: int,
        remaining_resource_demand: List[int],
        resource_timelines,
    ) -> float:
        if not self.instance.job_resource_requirements[job_id]:
            return 0.0

        total_pressure = 0.0
        duration = self.instance.processing[job_id]
        for resource_id, required_amount in self.instance.job_resource_requirements[job_id]:
            remaining_capacity = self._remaining_resource_capacity(resource_timelines[resource_id])
            if remaining_capacity <= 0:
                return float("inf")
            remaining_demand = remaining_resource_demand[resource_id]
            job_demand = duration * required_amount
            total_pressure += remaining_demand / remaining_capacity
            total_pressure += job_demand / remaining_capacity
        return total_pressure

    def _select_beam_critical_job(
        self,
        state: ScheduleState,
        ready_jobs: Set[int],
        remaining_resource_demand: List[int],
        resource_timelines,
    ) -> Optional[int]:
        if not ready_jobs:
            return None
        return max(
            ready_jobs,
            key=lambda job_id: (
                self._job_resource_pressure(job_id, remaining_resource_demand, resource_timelines),
                self._job_machine_regret(state, job_id, resource_timelines),
                -len(self.instance.eligible[job_id]),
                self.instance.processing[job_id],
                -job_id,
            ),
        )

    def _job_machine_regret(
        self,
        state: ScheduleState,
        job_id: int,
        resource_timelines,
    ) -> int:
        predecessor_finish = self._predecessor_finish(state, job_id)
        finishes: List[int] = []
        for machine_id in self.instance.eligible[job_id]:
            feasible_start = max(predecessor_finish, self._earliest_machine_ready(state, job_id, machine_id))
            while True:
                updated_start = feasible_start
                for resource_id, required_amount in self.instance.job_resource_requirements[job_id]:
                    candidate_start = resource_timelines[resource_id].earliest_feasible_start(
                        updated_start,
                        self.instance.processing[job_id],
                        required_amount,
                    )
                    if candidate_start >= 10**9:
                        updated_start = 10**9
                        break
                    updated_start = max(updated_start, candidate_start)
                if updated_start >= 10**9:
                    feasible_start = 10**9
                    break
                if updated_start == feasible_start:
                    break
                feasible_start = updated_start
            if feasible_start < 10**9:
                finishes.append(feasible_start + self.instance.processing[job_id])
        if len(finishes) < 2:
            return 10**9
        finishes.sort()
        return finishes[1] - finishes[0]

    def _remaining_resource_demand(self, unscheduled_jobs: Set[int]) -> List[int]:
        demand = [0] * self.instance.n_resources
        for job_id in unscheduled_jobs:
            duration = self.instance.processing[job_id]
            for resource_id, required_amount in self.instance.job_resource_requirements[job_id]:
                demand[resource_id] += duration * required_amount
        return demand

    def _remaining_resource_capacity(self, timeline) -> int:
        return sum(
            max(0, seg_end - seg_start) * seg_capacity
            for seg_start, seg_end, seg_capacity in timeline.segments
            if seg_capacity > 0 and seg_end > seg_start
        )

    def _score_beam_node(
        self,
        state: ScheduleState,
        unscheduled_jobs: Set[int],
        resource_timelines,
    ) -> Tuple[float, int, int, int]:
        remaining_resource_demand = self._remaining_resource_demand(unscheduled_jobs)
        total_pressure = 0.0
        for resource_id in range(self.instance.n_resources):
            remaining_capacity = self._remaining_resource_capacity(resource_timelines[resource_id])
            if remaining_capacity <= 0:
                total_pressure += float(remaining_resource_demand[resource_id] > 0) * 10**9
            elif remaining_resource_demand[resource_id] > 0:
                total_pressure += remaining_resource_demand[resource_id] / remaining_capacity

        pressured_window = self._find_windowed_resource_pressure(
            state,
            unscheduled_jobs,
            resource_timelines,
        )
        overload_penalty = 0
        if pressured_window is not None:
            _, _, _, demand, capacity = pressured_window
            overload_penalty = max(0, demand - capacity)

        projected_makespan = max((finish for finish in state.finish_times if finish >= 0), default=0)
        return (overload_penalty, total_pressure, projected_makespan, len(unscheduled_jobs))

    def _estimated_earliest_job_start(self, state: ScheduleState, job_id: int) -> int:
        predecessor_finish = self._predecessor_finish(state, job_id)
        earliest_machine_start = min(
            self._earliest_machine_ready(state, job_id, machine_id)
            for machine_id in self.instance.eligible[job_id]
        )
        return max(predecessor_finish, earliest_machine_start)

    def _windowed_remaining_resource_demand(
        self,
        state: ScheduleState,
        unscheduled_jobs: Set[int],
        resource_timelines,
    ) -> Dict[int, List[Tuple[int, int, int, int]]]:
        windowed_demand: Dict[int, List[Tuple[int, int, int, int]]] = {}
        for resource_id in range(self.instance.n_resources):
            windows = [
                (seg_start, seg_end, 0, max(0, seg_end - seg_start) * seg_capacity)
                for seg_start, seg_end, seg_capacity in resource_timelines[resource_id].segments
                if seg_capacity > 0 and seg_end > seg_start
            ]
            windowed_demand[resource_id] = windows

        for job_id in unscheduled_jobs:
            estimated_start = self._estimated_earliest_job_start(state, job_id)
            estimated_finish = estimated_start + self.instance.processing[job_id]
            for resource_id, required_amount in self.instance.job_resource_requirements[job_id]:
                windows = windowed_demand[resource_id]
                for idx, (seg_start, seg_end, current_demand, capacity) in enumerate(windows):
                    overlap_start = max(seg_start, estimated_start)
                    overlap_end = min(seg_end, estimated_finish)
                    if overlap_start >= overlap_end:
                        continue
                    added_demand = (overlap_end - overlap_start) * required_amount
                    windows[idx] = (seg_start, seg_end, current_demand + added_demand, capacity)
        return windowed_demand

    def _find_windowed_resource_pressure(
        self,
        state: ScheduleState,
        unscheduled_jobs: Set[int],
        resource_timelines,
    ) -> Optional[Tuple[int, int, int, int, int]]:
        worst_window: Optional[Tuple[int, int, int, int, int]] = None
        windowed_demand = self._windowed_remaining_resource_demand(state, unscheduled_jobs, resource_timelines)
        for resource_id, windows in windowed_demand.items():
            for window_start, window_end, demand, capacity in windows:
                if demand <= capacity:
                    continue
                overload = demand - capacity
                if worst_window is None or overload > (worst_window[3] - worst_window[4]):
                    worst_window = (resource_id, window_start, window_end, demand, capacity)
        return worst_window

    def _find_aggregate_resource_impossibility(
        self,
        remaining_resource_demand: List[int],
        resource_timelines,
    ) -> Optional[Tuple[int, int, int]]:
        for resource_id in range(self.instance.n_resources):
            remaining_capacity = self._remaining_resource_capacity(resource_timelines[resource_id])
            remaining_demand = remaining_resource_demand[resource_id]
            if remaining_demand > remaining_capacity:
                return resource_id, remaining_demand, remaining_capacity
        return None

    def _blocked_job_diagnostics(self, blocked_job_ids: List[int]) -> str:
        labels: List[str] = []
        for job_id in blocked_job_ids:
            if self._job_is_individually_placeable(job_id):
                labels.append(str(job_id + 1))
            else:
                labels.append(f"{job_id + 1}(!solo)")
        return ",".join(labels)

    def _job_is_individually_placeable(self, job_id: int) -> bool:
        if not self.instance.eligible[job_id]:
            return False
        duration = self.instance.processing[job_id]
        for resource_id, required_amount in self.instance.job_resource_requirements[job_id]:
            if not any(
                capacity >= required_amount and (period_end - period_start) >= duration
                for period_start, period_end, capacity in self.instance.resource_periods[resource_id]
            ):
                return False
        return True

    def _predecessor_finish(self, state: ScheduleState, job_id: int) -> int:
        return max(
            (state.finish_times[pred_id] for pred_id in self.instance.predecessor_indices[job_id]),
            default=0,
        )

    def _earliest_machine_ready(self, state: ScheduleState, job_id: int, machine_id: int) -> int:
        sequence = state.machine_sequences[machine_id]
        if not sequence:
            return self.instance.initial_setup[job_id]
        previous_job_id = sequence[-1]
        return state.finish_times[previous_job_id] + self.instance.setup[job_id][previous_job_id]

    def _machine_insertions(
        self, state: ScheduleState, job_id: int, machine_id: int, predecessor_finish: int
    ) -> List[Tuple[int, int, int, int]]:
        sequence = state.machine_sequences[machine_id]
        processing_time = self.instance.processing[job_id]
        candidates: List[Tuple[int, int, int, int]] = []

        for insert_pos in range(len(sequence) + 1):
            prev_job_id = sequence[insert_pos - 1] if insert_pos > 0 else None
            next_job_id = sequence[insert_pos] if insert_pos < len(sequence) else None

            if prev_job_id is None:
                earliest_start = max(predecessor_finish, self.instance.initial_setup[job_id])
            else:
                earliest_start = max(
                    predecessor_finish,
                    state.finish_times[prev_job_id] + self.instance.setup[job_id][prev_job_id],
                )

            if next_job_id is None:
                latest_finish = earliest_start + processing_time
                slack_after = 0
            else:
                latest_finish = state.start_times[next_job_id] - self.instance.setup[next_job_id][job_id]
                slack_after = latest_finish - (earliest_start + processing_time)

            finish = earliest_start + processing_time
            if finish <= latest_finish:
                candidates.append((insert_pos, earliest_start, finish, max(0, slack_after)))

        candidates.sort(key=lambda item: (item[1], item[3], item[0]))
        return candidates[:4]

    def _resource_feasible_with_penalty(
        self,
        resource_usage: List[List[Tuple[int, int, int, int]]],
        job_id: int,
        start: int,
        finish: int,
    ) -> Tuple[bool, int]:
        penalty = 0
        for resource_id, required_capacity in self.instance.job_resource_requirements[job_id]:
            usage_intervals = resource_usage[resource_id]
            periods = self.instance.resource_periods[resource_id]
            boundaries = {start, finish}

            overlapping_periods = []
            for period_start, period_end, capacity in periods:
                if period_start < finish and start < period_end:
                    clipped_start = max(start, period_start)
                    clipped_end = min(finish, period_end)
                    overlapping_periods.append((period_start, period_end, capacity))
                    boundaries.add(clipped_start)
                    boundaries.add(clipped_end)

            if not overlapping_periods:
                return False, 10**9

            overlapping_usage = []
            for used_start, used_end, used_capacity, _ in usage_intervals:
                if used_start < finish and start < used_end:
                    clipped_start = max(start, used_start)
                    clipped_end = min(finish, used_end)
                    overlapping_usage.append((used_start, used_end, used_capacity))
                    boundaries.add(clipped_start)
                    boundaries.add(clipped_end)

            sorted_boundaries = sorted(boundaries)
            for idx in range(len(sorted_boundaries) - 1):
                seg_start = sorted_boundaries[idx]
                seg_end = sorted_boundaries[idx + 1]
                if seg_start >= seg_end:
                    continue

                available_capacity = 0
                for period_start, period_end, capacity in overlapping_periods:
                    if period_start < seg_end and seg_start < period_end:
                        available_capacity = max(available_capacity, capacity)

                used_capacity = 0
                for used_start, used_end, capacity in overlapping_usage:
                    if used_start < seg_end and seg_start < used_end:
                        used_capacity += capacity

                if used_capacity + required_capacity > available_capacity:
                    return False, 10**9
                penalty += max(0, required_capacity * 20 - (available_capacity - used_capacity - required_capacity))

        return True, penalty

    def _machine_flexibility_penalty(self, job_id: int, machine_id: int) -> int:
        flexibility = len(self.instance.eligible[job_id])
        if flexibility <= 1:
            return 0
        machine_pressure = self.instance.machine_job_counts[machine_id]
        return max(0, 20 * flexibility - machine_pressure)

    def _apply_candidate(self, state: ScheduleState, candidate: CandidateMove) -> bool:
        job_id = candidate.job_id
        machine_id = candidate.machine_id
        state.machine_sequences[machine_id].insert(candidate.insert_pos, job_id)
        state.machine_assignment[job_id] = machine_id
        state.start_times[job_id] = candidate.start
        state.finish_times[job_id] = candidate.finish
        state.scheduled_jobs.add(job_id)
        state.insertion_order.append(job_id)

        for resource_id, required_capacity in self.instance.job_resource_requirements[job_id]:
            state.resource_usage[resource_id].append((candidate.start, candidate.finish, required_capacity, job_id))

        self._refresh_machine_tail(state, machine_id)
        return True

    def _rollback_conflict_neighborhood(
        self,
        state: ScheduleState,
        unscheduled_jobs: Set[int],
        ready_jobs: Set[int],
    ) -> List[int]:
        if not state.insertion_order:
            return []

        blocked_jobs = sorted(ready_jobs, key=lambda job_id: self._dynamic_job_priority_key(state, job_id))[:3]
        self.rng.shuffle(blocked_jobs)
        removal_candidates: List[int] = []
        seen: Set[int] = set()

        for blocked_job_id in blocked_jobs:
            for machine_id in self.instance.eligible[blocked_job_id]:
                for scheduled_job_id in reversed(state.machine_sequences[machine_id]):
                    if scheduled_job_id not in seen:
                        removal_candidates.append(scheduled_job_id)
                        seen.add(scheduled_job_id)
                    if len(removal_candidates) >= self.rollback_size:
                        break
                if len(removal_candidates) >= self.rollback_size:
                    break

            if len(removal_candidates) < self.rollback_size:
                for resource_id, _, _, _ in self._blocked_resource_intervals(state, blocked_job_id):
                    resource_intervals = list(reversed(state.resource_usage[resource_id]))
                    self.rng.shuffle(resource_intervals)
                    for _, _, _, used_job_id in resource_intervals:
                        if used_job_id not in seen:
                            removal_candidates.append(used_job_id)
                            seen.add(used_job_id)
                        if len(removal_candidates) >= self.rollback_size:
                            break
                    if len(removal_candidates) >= self.rollback_size:
                        break

        for scheduled_job_id in reversed(state.insertion_order):
            if len(removal_candidates) >= self.rollback_size:
                break
            if scheduled_job_id not in seen:
                removal_candidates.append(scheduled_job_id)
                seen.add(scheduled_job_id)

        removed_jobs = removal_candidates[: self.rollback_size]
        if not removed_jobs:
            return []

        self._remove_jobs(state, unscheduled_jobs, removed_jobs)
        return removed_jobs

    def _blocked_resource_intervals(
        self, state: ScheduleState, job_id: int
    ) -> List[Tuple[int, int, int, int]]:
        predecessor_finish = self._predecessor_finish(state, job_id)
        blocked = []
        for machine_id in self.instance.eligible[job_id]:
            start = self._earliest_machine_ready(state, job_id, machine_id)
            start = max(start, predecessor_finish)
            finish = start + self.instance.processing[job_id]
            for resource_id, required_capacity in self.instance.job_resource_requirements[job_id]:
                blocked.append((resource_id, start, finish, required_capacity))
        return blocked

    def _destroy_large_neighborhood(
        self,
        state: ScheduleState,
        unscheduled_jobs: Set[int],
        blocked_jobs: List[int],
    ) -> List[int]:
        if not state.insertion_order:
            return []

        target_size = max(self.rollback_size, min(24, self.rollback_size * 3))
        removal_candidates: List[int] = []
        seen: Set[int] = set()

        prioritized_blocked = blocked_jobs[: min(4, len(blocked_jobs))]
        for blocked_job_id in prioritized_blocked:
            for machine_id in self.instance.eligible[blocked_job_id]:
                for scheduled_job_id in reversed(state.machine_sequences[machine_id]):
                    if scheduled_job_id not in seen:
                        removal_candidates.append(scheduled_job_id)
                        seen.add(scheduled_job_id)
                    if len(removal_candidates) >= target_size:
                        break
                if len(removal_candidates) >= target_size:
                    break

            if len(removal_candidates) < target_size:
                for resource_id, start, finish, _ in self._blocked_resource_intervals(state, blocked_job_id):
                    for used_start, used_end, _, used_job_id in reversed(state.resource_usage[resource_id]):
                        if used_job_id in seen:
                            continue
                        if used_start < finish and start < used_end:
                            removal_candidates.append(used_job_id)
                            seen.add(used_job_id)
                        if len(removal_candidates) >= target_size:
                            break
                    if len(removal_candidates) >= target_size:
                        break

        for scheduled_job_id in reversed(state.insertion_order):
            if len(removal_candidates) >= target_size:
                break
            if scheduled_job_id not in seen:
                removal_candidates.append(scheduled_job_id)
                seen.add(scheduled_job_id)

        removed_jobs = removal_candidates[:target_size]
        if not removed_jobs:
            return []

        self._remove_jobs(state, unscheduled_jobs, removed_jobs)
        return removed_jobs

    def _remove_jobs(self, state: ScheduleState, unscheduled_jobs: Set[int], removed_jobs: List[int]) -> None:
        removed_set = self._expand_removed_jobs_with_scheduled_successors(state, set(removed_jobs))
        impacted_machines = {
            state.machine_assignment[job_id]
            for job_id in removed_set
            if state.machine_assignment[job_id] != 0
        }

        # Keep surviving partial sequences as machine prefixes. Removing jobs only from the
        # middle of a machine sequence can leave a cross-machine deadlock during partial rebuild.
        for machine_id in impacted_machines:
            sequence = state.machine_sequences[machine_id]
            first_removed_idx = next(
                (idx for idx, scheduled_job_id in enumerate(sequence) if scheduled_job_id in removed_set),
                None,
            )
            if first_removed_idx is None:
                continue
            removed_set.update(sequence[first_removed_idx:])

        for job_id in removed_set:
            unscheduled_jobs.add(job_id)
            machine_id = state.machine_assignment[job_id]
            if machine_id != 0:
                state.machine_sequences[machine_id] = [
                    scheduled_job_id
                    for scheduled_job_id in state.machine_sequences[machine_id]
                    if scheduled_job_id != job_id
                ]
                self._refresh_machine_tail(state, machine_id)
            state.machine_assignment[job_id] = 0
            state.start_times[job_id] = -1
            state.finish_times[job_id] = -1
            state.scheduled_jobs.discard(job_id)

        state.insertion_order = [
            job_id for job_id in state.insertion_order
            if job_id not in removed_set
        ]
        for resource_id in range(self.instance.n_resources):
            state.resource_usage[resource_id] = [
                interval
                for interval in state.resource_usage[resource_id]
                if interval[3] not in removed_set
            ]
        self._rebuild_state_from_sequences(state)

    def _refresh_machine_tail(self, state: ScheduleState, machine_id: int) -> None:
        sequence = state.machine_sequences[machine_id]
        if not sequence:
            state.machine_ready_time[machine_id] = 0
            state.previous_job_by_machine[machine_id] = None
            return
        last_job_id = sequence[-1]
        state.machine_ready_time[machine_id] = state.finish_times[last_job_id]
        state.previous_job_by_machine[machine_id] = last_job_id

    def _recompute_frontier(
        self, unscheduled_jobs: Set[int], scheduled_jobs: Set[int]
    ) -> Tuple[List[int], Set[int]]:
        remaining_predecessors = [0] * self.instance.n_jobs
        ready_jobs: Set[int] = set()
        for job_id in unscheduled_jobs:
            missing = sum(
                1
                for predecessor_id in self.instance.predecessor_indices[job_id]
                if predecessor_id not in scheduled_jobs
            )
            remaining_predecessors[job_id] = missing
            if missing == 0:
                ready_jobs.add(job_id)
        return remaining_predecessors, ready_jobs

    def _rollback_recent_large_state(
        self,
        state: ScheduleState,
        unscheduled_jobs: Set[int],
    ) -> List[int]:
        if not state.insertion_order:
            return []
        remove_count = min(len(state.insertion_order), max(self.rollback_size, self.frontier_size // 2))
        removed_jobs = list(reversed(state.insertion_order[-remove_count:]))
        self._remove_jobs(state, unscheduled_jobs, removed_jobs)
        return removed_jobs

    def _rollback_blocked_large_state(
        self,
        state: ScheduleState,
        unscheduled_jobs: Set[int],
        blocked_ready_jobs: List[int],
    ) -> List[int]:
        if not state.insertion_order:
            return []

        removal_candidates: List[int] = []
        seen: Set[int] = set()
        target_size = min(len(state.insertion_order), max(self.rollback_size * 2, self.frontier_size))

        for blocked_job_id in blocked_ready_jobs[: min(5, len(blocked_ready_jobs))]:
            conflict_jobs = self._conflict_jobs_for_blocked_job(state, blocked_job_id)
            for scheduled_job_id in conflict_jobs:
                if scheduled_job_id not in seen:
                    seen.add(scheduled_job_id)
                    removal_candidates.append(scheduled_job_id)
                if len(removal_candidates) >= target_size:
                    break
            if len(removal_candidates) >= target_size:
                break

            for machine_id in self.instance.eligible[blocked_job_id]:
                for scheduled_job_id in reversed(state.machine_sequences[machine_id]):
                    if scheduled_job_id not in seen:
                        seen.add(scheduled_job_id)
                        removal_candidates.append(scheduled_job_id)
                    if len(removal_candidates) >= target_size:
                        break
                if len(removal_candidates) >= target_size:
                    break

            if len(removal_candidates) < target_size:
                for predecessor_id in self.instance.predecessor_indices[blocked_job_id]:
                    if predecessor_id in state.scheduled_jobs and predecessor_id not in seen:
                        seen.add(predecessor_id)
                        removal_candidates.append(predecessor_id)
                    if len(removal_candidates) >= target_size:
                        break

            if len(removal_candidates) < target_size:
                for resource_id, _ in self.instance.job_resource_requirements[blocked_job_id]:
                    relevant_jobs = [
                        job_id
                        for job_id in reversed(state.insertion_order)
                        if job_id not in seen
                        and any(req_resource_id == resource_id for req_resource_id, _ in self.instance.job_resource_requirements[job_id])
                    ]
                    for scheduled_job_id in relevant_jobs[: max(2, self.rollback_size // 2)]:
                        seen.add(scheduled_job_id)
                        removal_candidates.append(scheduled_job_id)
                        if len(removal_candidates) >= target_size:
                            break
                    if len(removal_candidates) >= target_size:
                        break

            if len(removal_candidates) >= target_size:
                break

        for scheduled_job_id in reversed(state.insertion_order):
            if len(removal_candidates) >= target_size:
                break
            if scheduled_job_id not in seen:
                seen.add(scheduled_job_id)
                removal_candidates.append(scheduled_job_id)

        if not removal_candidates:
            return []

        self._remove_jobs(state, unscheduled_jobs, removal_candidates[:target_size])
        return removal_candidates[:target_size]

    def _rollback_resource_pressure_state(
        self,
        state: ScheduleState,
        unscheduled_jobs: Set[int],
        resource_id: int,
        window_start: int,
        window_end: int,
    ) -> List[int]:
        if not state.insertion_order:
            return []

        candidates: List[int] = []
        seen: Set[int] = set()
        target_size = min(len(state.insertion_order), max(self.rollback_size, self.frontier_size // 2))

        overlapping_usage = [
            (used_start, used_end, required_capacity, job_id)
            for used_start, used_end, required_capacity, job_id in state.resource_usage[resource_id]
            if used_start < window_end and window_start < used_end
        ]
        if not overlapping_usage:
            overlapping_usage = state.resource_usage[resource_id]

        for _, _, required_capacity, job_id in sorted(
            overlapping_usage,
            key=lambda item: (
                item[2] * max(0, min(item[1], window_end) - max(item[0], window_start)),
                item[2],
                item[1] - item[0],
                item[1],
            ),
            reverse=True,
        ):
            if job_id in seen:
                continue
            seen.add(job_id)
            candidates.append(job_id)
            if len(candidates) >= target_size:
                break

        for predecessor_id in list(candidates):
            for pred in self.instance.predecessor_indices[predecessor_id]:
                if pred in state.scheduled_jobs and pred not in seen:
                    seen.add(pred)
                    candidates.append(pred)
                if len(candidates) >= target_size:
                    break
            if len(candidates) >= target_size:
                break

        if not candidates:
            return []

        removed_jobs = candidates[:target_size]
        self._remove_jobs(state, unscheduled_jobs, removed_jobs)
        return removed_jobs

    def _conflict_jobs_for_blocked_job(self, state: ScheduleState, blocked_job_id: int) -> List[int]:
        candidate_jobs: List[int] = []
        seen: Set[int] = set()

        for resource_id, start, finish, _ in self._blocked_resource_intervals(state, blocked_job_id):
            overlaps = [
                (used_capacity, used_end - used_start, used_end, used_job_id)
                for used_start, used_end, used_capacity, used_job_id in state.resource_usage[resource_id]
                if used_job_id not in seen and used_start < finish and start < used_end
            ]
            overlaps.sort(reverse=True)
            for _, _, _, used_job_id in overlaps[: max(2, self.rollback_size // 2)]:
                seen.add(used_job_id)
                candidate_jobs.append(used_job_id)

        for predecessor_id in self.instance.predecessor_indices[blocked_job_id]:
            if predecessor_id in state.scheduled_jobs and predecessor_id not in seen:
                seen.add(predecessor_id)
                candidate_jobs.append(predecessor_id)

        return candidate_jobs

    def _build_resource_timelines_from_state(self, state: ScheduleState):
        resource_timelines = self.instance._build_resource_timelines()
        for job_id in state.scheduled_jobs:
            for resource_id, required_amount in self.instance.job_resource_requirements[job_id]:
                resource_timelines[resource_id].commit(
                    state.start_times[job_id],
                    state.finish_times[job_id],
                    required_amount,
                )
        return resource_timelines

    def _expand_removed_jobs_with_scheduled_successors(
        self,
        state: ScheduleState,
        removed_jobs: Set[int],
    ) -> Set[int]:
        closure = set(removed_jobs)
        stack = list(removed_jobs)
        while stack:
            job_id = stack.pop()
            for successor_id in self.instance.successor_indices[job_id]:
                if successor_id in state.scheduled_jobs and successor_id not in closure:
                    closure.add(successor_id)
                    stack.append(successor_id)
        return closure

    def _rebuild_state_from_sequences(self, state: ScheduleState) -> None:
        feasible, start_times, finish_times, machine_assignment = self.instance.schedule_partial_sequences(
            state.machine_sequences
        )
        if not feasible:
            raise RuntimeError("Partial schedule became infeasible during destroy/repair rebuild.")

        state.machine_assignment = machine_assignment
        state.start_times = start_times
        state.finish_times = finish_times
        state.scheduled_jobs = {
            job_id
            for machine_id in range(1, self.instance.n_machines + 1)
            for job_id in state.machine_sequences[machine_id]
        }
        state.insertion_order = [job_id for job_id in state.insertion_order if job_id in state.scheduled_jobs]
        state.resource_usage = [[] for _ in range(self.instance.n_resources)]
        for job_id in state.scheduled_jobs:
            for resource_id, required_capacity in self.instance.job_resource_requirements[job_id]:
                state.resource_usage[resource_id].append(
                    (state.start_times[job_id], state.finish_times[job_id], required_capacity, job_id)
                )

        state.machine_ready_time = [0] * (self.instance.n_machines + 1)
        state.previous_job_by_machine = [None] * (self.instance.n_machines + 1)
        for machine_id in range(1, self.instance.n_machines + 1):
            self._refresh_machine_tail(state, machine_id)


def initial_feasible_solution(instance: PMSInstance, **kwargs) -> Dict:
    total_time_limit_s = kwargs.pop("total_time_limit_s", 180)
    allow_a2_fallback = kwargs.pop("allow_a2_fallback", True)
    fast_large_instance_mode = kwargs.pop("fast_large_instance_mode", True)
    diagnostics_enabled = kwargs.pop("diagnostics_enabled", False)
    attempt_configs = [
        {"frontier_size": 12, "rollback_size": 8, "max_rollbacks": 200},
        {"frontier_size": 20, "rollback_size": 12, "max_rollbacks": 350},
        {"frontier_size": 28, "rollback_size": 16, "max_rollbacks": 500},
        {"frontier_size": 36, "rollback_size": 20, "max_rollbacks": 700},
    ]

    last_error: Optional[Exception] = None
    diagnostic_messages: List[str] = []
    base_seed = kwargs.pop("random_seed", 0)
    deadline = time.time() + total_time_limit_s if total_time_limit_s is not None else None

    if fast_large_instance_mode and instance.n_jobs >= 500:
        beam_width = kwargs.get("beam_width", 4)
        beam_branch_limit = kwargs.get("beam_branch_limit", 3)
        builder = InitialSolutionBuilder(
            instance,
            frontier_size=kwargs.get("frontier_size", 8),
            rollback_size=kwargs.get("rollback_size", 4),
            max_rollbacks=kwargs.get("max_rollbacks", 40),
            random_seed=base_seed,
        )
        try:
            return builder.build_large_instance_feasible_solution(deadline)
        except RuntimeError as exc:
            last_error = exc
            diagnostic_messages.append(f"large-decode failed: {exc}")

        if deadline is None or time.time() < deadline:
            builder = InitialSolutionBuilder(
                instance,
                frontier_size=kwargs.get("frontier_size", 8),
                rollback_size=kwargs.get("rollback_size", 4),
                max_rollbacks=kwargs.get("max_rollbacks", 40),
                random_seed=base_seed,
            )
            try:
                return builder.beam_search_large_instance(
                    deadline,
                    beam_width=beam_width,
                    branch_limit=beam_branch_limit,
                )
            except RuntimeError as exc:
                last_error = exc
                diagnostic_messages.append(
                    f"beam-search[width={beam_width},branch_limit={beam_branch_limit}] failed: {exc}"
                )

        fast_stages = [
            {"max_restarts": 60, "candidate_pool_size": 2, "ready_job_limit": 8},
            {"max_restarts": 90, "candidate_pool_size": 3, "ready_job_limit": 10},
            {"max_restarts": 120, "candidate_pool_size": 4, "ready_job_limit": 12},
        ]
        for stage_idx, stage in enumerate(fast_stages):
            if deadline is not None and time.time() >= deadline:
                break
            builder = InitialSolutionBuilder(
                instance,
                frontier_size=kwargs.get("frontier_size", 8),
                rollback_size=kwargs.get("rollback_size", 4),
                max_rollbacks=kwargs.get("max_rollbacks", 40),
                random_seed=base_seed + stage_idx,
            )
            try:
                return builder.randomized_append_solution(deadline=deadline, **stage)
            except RuntimeError as exc:
                last_error = exc
                diagnostic_messages.append(
                    "randomized-append"
                    f"[stage={stage_idx + 1},restarts={stage['max_restarts']},pool={stage['candidate_pool_size']},"
                    f"ready_limit={stage['ready_job_limit']}] failed: {exc}"
                )

        if diagnostics_enabled and diagnostic_messages:
            raise RuntimeError(" | ".join(diagnostic_messages))
        raise RuntimeError(str(last_error) if last_error else "Could not build a feasible initial solution.")

    for attempt_idx, config in enumerate(attempt_configs):
        if deadline is not None and time.time() >= deadline:
            break
        merged = dict(config)
        merged.update(kwargs)
        merged["random_seed"] = base_seed + attempt_idx
        builder = InitialSolutionBuilder(instance, **merged)
        try:
            return builder.repaired_initial_feasible_solution(deadline)
        except RuntimeError as exc:
            last_error = exc
            if diagnostics_enabled:
                diagnostic_messages.append(
                    "repair"
                    f"[attempt={attempt_idx + 1},frontier={merged['frontier_size']},rollback={merged['rollback_size']},"
                    f"max_rollbacks={merged['max_rollbacks']}] failed: {exc}"
                )

    fallback_pools = [3, 5, 8]
    for pool_idx, pool_size in enumerate(fallback_pools):
        if deadline is not None and time.time() >= deadline:
            break
        builder = InitialSolutionBuilder(
            instance,
            frontier_size=kwargs.get("frontier_size", 20),
            rollback_size=kwargs.get("rollback_size", 12),
            max_rollbacks=kwargs.get("max_rollbacks", 300),
            random_seed=base_seed + 100 + pool_idx,
        )
        try:
            return builder.randomized_append_solution(
                max_restarts=180 if instance.n_jobs >= 500 else 250,
                candidate_pool_size=pool_size,
                ready_job_limit=16 if instance.n_jobs >= 500 else 24,
                deadline=deadline,
            )
        except RuntimeError as exc:
            last_error = exc
            if diagnostics_enabled:
                diagnostic_messages.append(
                    "randomized-fallback"
                    f"[attempt={pool_idx + 1},pool={pool_size}] failed: {exc}"
                )

    a2_dir = Path(__file__).resolve().parent.parent / "A2"
    sa_module_path = a2_dir / "pms_simulated_annealing.py"
    if allow_a2_fallback and sa_module_path.exists() and (deadline is None or time.time() < deadline):
        import sys

        if str(a2_dir) not in sys.path:
            sys.path.insert(0, str(a2_dir))
        spec = importlib.util.spec_from_file_location("a2_pms_simulated_annealing", sa_module_path)
        if spec is not None and spec.loader is not None:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            sa = module.SimulatedAnnealing(module.PMSInstance(instance.json_path))
            for pool_size in (3, 5, 8):
                if deadline is not None and time.time() >= deadline:
                    break
                try:
                    sequences = sa.construct_feasible_solution(max_restarts=400, candidate_pool_size=pool_size)
                    objective, tardiness, makespan, feasible, solution, _, _, _ = instance.decode_sequences(sequences)
                    if feasible:
                        return {
                            "objective": objective,
                            "tardiness": tardiness,
                            "makespan": makespan,
                            "solution": solution,
                            "job_sequences_by_machine": sequences,
                        }
                except RuntimeError as exc:
                    last_error = exc
                    if diagnostics_enabled:
                        diagnostic_messages.append(f"a2-fallback[pool={pool_size}] failed: {exc}")

    if diagnostics_enabled and diagnostic_messages:
        raise RuntimeError(" | ".join(diagnostic_messages))
    raise RuntimeError(str(last_error) if last_error else "Could not build a feasible initial solution.")
