"""
Policy ladder: measures how well the reward separates good play from bad play.

Four scripted policies of increasing competence are run against the same task
specs. The reward should rank them L0 < L1 < L2 < L3, and — more importantly —
the L3-L2 gap should be large enough that a learner can tell a good episode
from a merely adequate one.

The simulation is fully seeded off task_spec["seed"] (simulation.py:61) and the
policies are deterministic, so every number here is exactly reproducible;
variation comes from running across several task specs, not from repeats.

Usage:
    uv run python policy_ladder.py              # full ladder
    uv run python policy_ladder.py --step-sizes # step-size invariance check
"""

import argparse
import statistics
from typing import Any, Callable, Dict, List, Optional

from models import (CraneStatus, DisruptionType, VesselStatus,
                    WIND_CRANE_HALT_KNOTS)
from scenarios import ALL_TASKS
from simulation import PortSimulation

# Tasks the ladder is measured on. Deliberately spans the three structural
# profiles: calm (8-9 vessels), storm (9-10 + two storms), and peak season
# (12-14 vessels), which has a very different berth-demand ratio.
LADDER_TASKS = [
    "port_train_000",  # calm_week, 9 vessels
    "port_train_001",  # calm_week, 8 vessels
    "port_train_005",  # storm_season, 9 vessels
    "port_train_006",  # storm_season, 10 vessels
    "port_train_010",  # labor_dispute, 9 vessels — where overtime can pay off
    "port_train_011",  # labor_dispute, 8 vessels
    "port_train_015",  # peak_season, 13 vessels
    "port_train_016",  # peak_season, 12 vessels
]

DEFAULT_STEP_HOURS = 6.0


# ---------------------------------------------------------------------------
# Shared action helpers (same patterns as test_portmanager.py's dispatch loops)
# ---------------------------------------------------------------------------

def _find_empty_berth(sim: PortSimulation, min_draft: float = 0.0):
    for b in sim.berths.values():
        if b.vessel_id is None and b.draft_m >= min_draft:
            return b
    return None


def _berth_waiting_vessels(sim: PortSimulation, tide_aware: bool = True) -> None:
    for vessel in list(sim.vessels.values()):
        if vessel.status != VesselStatus.WAITING:
            continue
        berth = _find_empty_berth(sim, vessel.draft_required_m)
        if berth is None:
            continue
        if tide_aware and vessel.is_deep_draft and not sim.tide_high:
            continue
        sim.assign_berth(vessel.vessel_id, berth.berth_id)


def _assign_local_cranes(sim: PortSimulation, up_to_max: bool = False) -> None:
    """Assign idle cranes already standing at the vessel's own berth."""
    for vessel in list(sim.vessels.values()):
        if vessel.status != VesselStatus.BERTHED or vessel.remaining_moves <= 0:
            continue
        berth = sim.berths.get(vessel.berth_id)
        if not berth:
            continue
        target = min(vessel.max_cranes, berth.max_cranes)
        if not up_to_max and len(vessel.cranes_assigned) >= vessel.min_cranes:
            continue
        for crane in sim.cranes.values():
            if (crane.berth_id == vessel.berth_id
                    and crane.status == CraneStatus.IDLE
                    and len(vessel.cranes_assigned) < target):
                sim.assign_cranes(vessel.vessel_id, [crane.crane_id])


def _relocate_idle_cranes(sim: PortSimulation) -> None:
    """Pull idle cranes off empty berths toward vessels that are short-handed."""
    for vessel in list(sim.vessels.values()):
        if vessel.status != VesselStatus.BERTHED or vessel.remaining_moves <= 0:
            continue
        if len(vessel.cranes_assigned) >= vessel.min_cranes:
            continue
        if sim.wind_speed_knots >= WIND_CRANE_HALT_KNOTS:
            continue
        for crane in list(sim.cranes.values()):
            if crane.status != CraneStatus.IDLE or crane.berth_id == vessel.berth_id:
                continue
            source = sim.berths.get(crane.berth_id)
            if source and source.vessel_id is None:
                sim.move_crane(crane.crane_id, vessel.berth_id)
                break  # one move per pass, mirrors the smart loop in the tests


def _set_yard_plans(sim: PortSimulation, segregate: bool = False) -> None:
    for vessel in list(sim.vessels.values()):
        if vessel.status != VesselStatus.BERTHED or vessel.yard_blocks_import:
            continue
        if segregate and vessel.reefer_count > 0:
            blocks = [b.block_id for b in sim.yard_blocks.values()
                      if b.has_power_points
                      and b.current_occupancy < b.effective_capacity][:2]
            if blocks:
                sim.set_yard_plan(vessel.vessel_id, blocks, "reefer")
        if segregate and vessel.hazmat_count > 0:
            blocks = [b.block_id for b in sim.yard_blocks.values()
                      if b.hazmat_zone
                      and b.current_occupancy < b.effective_capacity][:2]
            if blocks:
                sim.set_yard_plan(vessel.vessel_id, blocks, "hazmat")
        blocks = [b.block_id for b in sim.yard_blocks.values()
                  if b.current_occupancy < b.effective_capacity
                  and not b.hazmat_zone][:3]
        if blocks:
            sim.set_yard_plan(vessel.vessel_id, blocks, "dry")


def _yard_utilization(sim: PortSimulation) -> float:
    occ = sum(b.current_occupancy for b in sim.yard_blocks.values())
    cap = sum(b.effective_capacity for b in sim.yard_blocks.values())
    return occ / max(1, cap)


def _dispatch_trucks(sim: PortSimulation, per_gate: int = 30) -> None:
    """Drain the fullest yard blocks through the outbound gates.

    The rate scales with yard occupancy: gates always run, but emptying the
    yard is not free, since yard_efficiency wants occupancy in a 0.50-0.80
    band.
    """
    per_gate = max(5, int(per_gate * _yard_utilization(sim)))
    blocks = sorted(sim.yard_blocks.values(),
                    key=lambda b: b.current_occupancy, reverse=True)
    gates = [g for gid, g in sorted(sim.gate_lanes.items()) if gid.startswith("GO")]
    for gate, block in zip(gates, blocks):
        if block.current_occupancy <= 0:
            continue
        sim.dispatch_trucks(min(per_gate, block.current_occupancy),
                            block.block_id, gate.lane_id)


def _schedule_trains(sim: PortSimulation, lead_hours: float = 6.0,
                     drain_above: float = 0.55) -> None:
    """Book free tracks against the fullest blocks, when the yard needs it."""
    if _yard_utilization(sim) < drain_above:
        return
    for track in sim.rail_tracks.values():
        if track.scheduled_departure is not None:
            continue
        blocks = sorted(sim.yard_blocks.values(),
                        key=lambda b: b.current_occupancy, reverse=True)
        block_ids = [b.block_id for b in blocks[:3] if b.current_occupancy > 0]
        if not block_ids:
            continue
        # A departed track is bookable again once turned around.
        departure = max(sim.clock + lead_hours, track.available_from)
        if departure >= sim.planning_horizon:
            continue
        sim.schedule_train(track.track_id, block_ids, departure)


def _handle_disruptions(sim: PortSimulation) -> None:
    """Pay for overtime only where it buys back lost productivity.

    Overtime carries a cost, so it is worth it during a strike or breakdown
    (which suppress the crane rate) but not during a storm, which halts cranes
    outright no matter what you pay.
    """
    productivity_hits = {DisruptionType.LABOR_STRIKE,
                         DisruptionType.EQUIPMENT_BREAKDOWN}
    for d in sim.disruptions.values():
        if not d.active or d.resolved or d.agent_action is not None:
            continue
        action = "overtime" if d.disruption_type in productivity_hits else "accept"
        sim.handle_disruption(d.disruption_id, action)


# ---------------------------------------------------------------------------
# The ladder itself
# ---------------------------------------------------------------------------

def policy_l0(sim: PortSimulation) -> None:
    """Do nothing but let the clock run."""


def policy_l1(sim: PortSimulation) -> None:
    """Naive: berth whatever is waiting, put the minimum cranes on it."""
    _berth_waiting_vessels(sim)
    _assign_local_cranes(sim)


def policy_l2(sim: PortSimulation) -> None:
    """Decent: L1 plus yard plans, truck dispatch and rail departures."""
    _berth_waiting_vessels(sim)
    _assign_local_cranes(sim)
    _set_yard_plans(sim)
    _dispatch_trucks(sim)
    _schedule_trains(sim)


def policy_l3(sim: PortSimulation) -> None:
    """Careful: L2 plus crane relocation, cargo segregation, disruption response."""
    _berth_waiting_vessels(sim)
    _assign_local_cranes(sim, up_to_max=True)
    _relocate_idle_cranes(sim)
    _set_yard_plans(sim, segregate=True)
    _dispatch_trucks(sim)
    _schedule_trains(sim)
    _handle_disruptions(sim)


LADDER: List[tuple] = [
    ("L0 do-nothing", policy_l0),
    ("L1 naive", policy_l1),
    ("L2 decent", policy_l2),
    ("L3 careful", policy_l3),
]


def run_policy(task: Dict[str, Any], policy: Callable[[PortSimulation], None],
               step_hours: float = DEFAULT_STEP_HOURS) -> Dict[str, Any]:
    """Drive one episode exactly the way portmanager.advance_time does.

    Acts, advances, then records one step reward per advance — the same
    sequence the real tool performs (portmanager.py:351-354) — so the ladder
    measures the reward the agent would actually receive.
    """
    sim = PortSimulation(task)
    while sim.clock < sim.planning_horizon:
        policy(sim)
        sim.advance_to(min(sim.clock + step_hours, sim.planning_horizon))
        sim.step_rewards.append(sim.compute_step_reward())
    return sim.compute_final_reward()


def _tasks_by_id(ids: List[str]) -> List[Dict[str, Any]]:
    index = {t["id"]: t for t in ALL_TASKS["train"] + ALL_TASKS["test"]}
    missing = [i for i in ids if i not in index]
    if missing:
        raise SystemExit(f"unknown task ids: {missing}")
    return [index[i] for i in ids]


def run_ladder(task_ids: List[str], step_hours: float) -> Dict[str, List[float]]:
    tasks = _tasks_by_id(task_ids)
    results: Dict[str, List[float]] = {name: [] for name, _ in LADDER}

    header = f"{'task':18}{'scenario':16}{'nves':>5}" + "".join(
        f"{name:>16}" for name, _ in LADDER)
    print(header)
    print("-" * len(header))
    for task in tasks:
        row = f"{task['id']:18}{task['scenario_type']:16}{task['num_vessels']:>5}"
        for name, policy in LADDER:
            final = run_policy(task, policy, step_hours)
            results[name].append(final["total_reward"])
            row += f"{final['total_reward']:>16.4f}"
        print(row)

    print("-" * len(header))
    summary = f"{'mean':18}{'':16}{'':>5}"
    for name, _ in LADDER:
        summary += f"{statistics.mean(results[name]):>16.4f}"
    print(summary)
    return results


def report_separation(results: Dict[str, List[float]]) -> None:
    names = [n for n, _ in LADDER]
    print("\nSeparation between adjacent rungs (mean over tasks):")
    for lo, hi in zip(names, names[1:]):
        gap = statistics.mean(results[hi]) - statistics.mean(results[lo])
        flag = "" if gap > 0 else "   <-- NOT MONOTONIC"
        print(f"  {lo:16} -> {hi:16} {gap:+.4f}{flag}")

    l2, l3 = results["L2 decent"], results["L3 careful"]
    gap = statistics.mean(l3) - statistics.mean(l2)
    per_task = [b - a for a, b in zip(l2, l3)]
    wins = sum(1 for g in per_task if g > 0)
    print(f"\nHEADLINE  L3 - L2 = {gap:+.4f}  "
          f"(better on {wins}/{len(per_task)} tasks; "
          f"per-task {', '.join(f'{g:+.3f}' for g in per_task)})")


def run_fixed_cadence(task: Dict[str, Any], policy: Callable[[PortSimulation], None],
                      act_hours: float = 3.0, read_every: int = 1) -> Dict[str, Any]:
    """Identical play, different reward-read cadence.

    The policy acts on a fixed clock, so the episode unfolds identically; only
    how often compute_step_reward is called varies. Any difference in the final
    score is therefore a pure scoring artifact.
    """
    sim = PortSimulation(task)
    tick = 0
    while sim.clock < sim.planning_horizon:
        policy(sim)
        sim.advance_to(min(sim.clock + act_hours, sim.planning_horizon))
        tick += 1
        if tick % read_every == 0:
            sim.step_rewards.append(sim.compute_step_reward())
    return sim.compute_final_reward()


def check_step_sizes(task_ids: List[str]) -> None:
    """The score must not depend on how the agent slices time.

    Two separate questions, so two separate tables — conflating them is
    misleading, because a bigger step also means the policy acts less often.
    """
    tasks = _tasks_by_id(task_ids)

    print("A. SCORING ARTIFACT — identical play, reward read every N advances")
    cadences = [1, 2, 4]
    header = f"{'task':18}" + "".join(f"{f'read/{c}':>14}" for c in cadences) + f"{'spread':>10}"
    print(header)
    print("-" * len(header))
    spreads = []
    for task in tasks:
        row = f"{task['id']:18}"
        vals = []
        for c in cadences:
            final = run_fixed_cadence(task, policy_l2, read_every=c)
            vals.append(final["total_reward"])
            row += f"{final['total_reward']:>14.4f}"
        spreads.append(max(vals) - min(vals))
        print(row + f"{max(vals) - min(vals):>10.4f}")
    print("-" * len(header))
    print(f"mean spread from read cadence alone: {statistics.mean(spreads):.4f}"
          "   <- must be 0.0000\n")

    print("B. REAL BEHAVIOUR — bigger steps also mean acting less often")
    sizes = [3.0, 6.0, 12.0]
    header = f"{'task':18}" + "".join(f"{f'{s:.0f}h steps':>14}" for s in sizes) + f"{'spread':>10}"
    print(header)
    print("-" * len(header))
    spreads = []
    for task in tasks:
        row = f"{task['id']:18}"
        vals = []
        for s in sizes:
            final = run_policy(task, policy_l2, s)
            vals.append(final["total_reward"])
            row += f"{final['total_reward']:>14.4f}"
        spreads.append(max(vals) - min(vals))
        print(row + f"{max(vals) - min(vals):>10.4f}")
    print("-" * len(header))
    print(f"mean spread: {statistics.mean(spreads):.4f}"
          "   (legitimate — acting 56x beats acting 14x)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", nargs="*", default=LADDER_TASKS)
    ap.add_argument("--step-hours", type=float, default=DEFAULT_STEP_HOURS)
    ap.add_argument("--step-sizes", action="store_true",
                    help="run the step-size invariance check instead")
    args = ap.parse_args()

    if args.step_sizes:
        check_step_sizes(args.tasks)
        return

    results = run_ladder(args.tasks, args.step_hours)
    report_separation(results)


if __name__ == "__main__":
    main()
