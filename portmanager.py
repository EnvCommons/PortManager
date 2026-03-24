"""
PortManager environment for OpenReward.

A hyper-realistic container port terminal management environment where an agent
schedules cranes, berths, yard storage, and truck/rail departures over a 168-hour
planning horizon while handling disruptions (storms, labor strikes, customs delays,
equipment breakdowns, high-priority cargo).
"""

from typing import List

from pydantic import BaseModel

from openreward.environments import Environment, JSONObject, ToolOutput, tool, TextBlock
from simulation import PortSimulation, CRANE_BASELINE_MPH
from scenarios import ALL_TASKS
from models import PLANNING_HORIZON, REWARD_WEIGHTS, BERTH_CONFIGS


# ---------------------------------------------------------------------------
# Pydantic param models for tools
# ---------------------------------------------------------------------------

class ObservePortParams(BaseModel, extra="forbid"):
    """No parameters needed."""
    pass


class AssignBerthParams(BaseModel, extra="forbid"):
    vessel_id: str
    berth_id: str


class AssignCranesParams(BaseModel, extra="forbid"):
    vessel_id: str
    crane_ids: List[str]


class MoveCraneParams(BaseModel, extra="forbid"):
    crane_id: str
    berth_id: str


class SetYardPlanParams(BaseModel, extra="forbid"):
    vessel_id: str
    yard_block_ids: List[str]
    container_type: str


class DispatchTrucksParams(BaseModel, extra="forbid"):
    count: int
    yard_block_id: str
    gate_id: str


class ScheduleTrainParams(BaseModel, extra="forbid"):
    track_id: str
    yard_block_ids: List[str]
    departure_hour: float


class AdvanceTimeParams(BaseModel, extra="forbid"):
    hours: int


class HandleDisruptionParams(BaseModel, extra="forbid"):
    disruption_id: str
    action: str


class SubmitPlanParams(BaseModel, extra="forbid"):
    """End episode and compute final reward."""
    pass


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class PortManager(Environment):
    """Container port terminal management environment."""

    def __init__(self, task_spec: JSONObject, secrets: dict[str, str] = {}) -> None:
        super().__init__(task_spec)
        self.task_config = task_spec
        self.sim = PortSimulation(task_spec)
        self.finished = False

    @classmethod
    def list_tasks(cls, split: str) -> list[JSONObject]:
        if split not in ALL_TASKS:
            raise ValueError(f"Unknown split: {split}. Available: {list(ALL_TASKS.keys())}")
        return ALL_TASKS[split]

    @classmethod
    def list_splits(cls) -> list[str]:
        return ["train", "test"]

    async def get_prompt(self) -> List[TextBlock]:
        status = self.sim.get_port_status()

        # Build berth summary
        berth_lines = []
        for b in status["berths"]:
            state = f"OCCUPIED by {b['vessel_id']}" if b["occupied"] else "EMPTY"
            berth_lines.append(
                f"  - {b['berth_id']}: {b['length_m']}m, depth {b['draft_m']}m, "
                f"max {b['max_cranes']} cranes | {state}"
            )

        # Build crane summary
        crane_lines = []
        for c in status["cranes"]:
            line = f"  - {c['crane_id']}: berth {c['berth_id']}, {c['status']}"
            if c.get("vessel_id"):
                line += f", working {c['vessel_id']}"
            crane_lines.append(line)

        # Build vessel summary
        vessel_lines = []
        for v in status["vessels"]:
            line = (f"  - {v['vessel_id']}: {v['type']}, {v['teu_capacity']} TEU, "
                    f"status={v['status']}, draft={v['draft_required_m']}m")
            if v["status"] == "scheduled":
                line += f", arrives ~hr {v.get('actual_arrival', v['scheduled_arrival']):.1f}"
            elif v["status"] == "waiting":
                line += ", WAITING for berth"
            elif v["status"] == "berthed":
                line += (f", berth={v.get('berth_id')}, "
                        f"cranes={v.get('cranes_assigned', [])}, "
                        f"remaining={v['remaining_moves']} moves")
            if v["reefer_count"] > 0:
                line += f", {v['reefer_count']} reefer"
            if v["hazmat_count"] > 0:
                line += f", {v['hazmat_count']} hazmat"
            vessel_lines.append(line)

        # Disruption summary
        disruption_lines = []
        for d in status["disruptions"]:
            state = "ACTIVE" if d["active"] else ("resolved" if d["resolved"] else f"starts hr {d['start_hour']:.1f}")
            disruption_lines.append(
                f"  - {d['disruption_id']}: {d['type']}, severity {d['severity']:.0%}, {state}"
            )
        if not disruption_lines:
            disruption_lines.append("  (none scheduled)")

        prompt = f"""You are a container port terminal operations manager. Your goal is to efficiently manage vessel arrivals, crane assignments, yard storage, and cargo transportation over a {PLANNING_HORIZON:.0f}-hour (1-week) planning horizon.

## TERMINAL CONFIGURATION

### Berths ({len(BERTH_CONFIGS)} total)
{chr(10).join(berth_lines)}

### STS Quay Cranes (12 total, benchmark: {CRANE_BASELINE_MPH:.0f} moves/crane/hour)
{chr(10).join(crane_lines)}

### Yard (20 blocks, each 6 rows x 30 bays x 5 tiers = 630 effective TEU)
- Blocks YB01-YB04: have reefer power points
- Blocks YB19-YB20: hazmat segregation zones
- Current overall utilization: {status['yard_summary']['utilization_pct']}%

### Gates (4 inbound + 4 outbound lanes, 30 trucks/hr/lane)
### Rail (4 tracks, 120 TEU max per train)

## CURRENT STATE (hour {status['clock']:.1f} / {status['planning_horizon']:.0f})

Wind: {status['wind_speed_knots']:.0f} knots | Tide: {'HIGH (deep-draft ok)' if status['tide_high'] else 'LOW (deep-draft restricted)'}
Vessels: {status['vessels_departed']} departed / {status['vessels_total']} total

## VESSELS
{chr(10).join(vessel_lines)}

## DISRUPTIONS
{chr(10).join(disruption_lines)}

## AVAILABLE TOOLS

1. **observe_port()** - Get full snapshot of berths, cranes, yard, gates, rail, weather, disruptions
2. **assign_berth(vessel_id, berth_id)** - Dock a waiting vessel. Validates draft depth, berth availability, and tide (deep-draft vessels need high tide).
3. **assign_cranes(vessel_id, crane_ids)** - Assign idle cranes to a berthed vessel. Cranes must be at the same berth.
4. **move_crane(crane_id, berth_id)** - Relocate a crane to a different berth (takes 1 hour). Blocked during storms.
5. **set_yard_plan(vessel_id, yard_block_ids, container_type)** - Designate yard blocks for containers. Reefer needs power blocks (YB01-04), hazmat needs hazmat zones (YB19-20).
6. **dispatch_trucks(count, yard_block_id, gate_id)** - Route trucks between yard and gate for pickup/delivery.
7. **schedule_train(track_id, yard_block_ids, departure_hour)** - Schedule a rail departure with containers from designated blocks.
8. **advance_time(hours)** - Advance simulation by 1-12 hours. Processes all events, crane operations, gate throughput. Returns per-step reward.
9. **handle_disruption(disruption_id, action)** - Respond to active disruptions. Actions: "accept", "overtime" (+50% productivity, cost penalty), "reroute", "delay".
10. **submit_plan()** - End episode and compute final reward.

## REWARD

You are rewarded for keeping berths occupied, maintaining high crane productivity, completing vessels promptly, using yard space efficiently, minimizing truck wait times, filling trains before departure, and maintaining safety compliance. You are penalized for idle resources, vessel delays, safety violations, and poor utilization of gates and rail.

## KEY CONSTRAINTS

- Deep-draft vessels (>14m) can ONLY berth/depart during HIGH TIDE windows (~4h each, 2x/day)
- Cranes cannot move during storms (wind >= 40 knots)
- Storms halt crane operations and reduce gate throughput 50%
- Labor strikes reduce productivity to 0-50%
- Reefer containers MUST go to power blocks (YB01-04)
- Hazmat containers MUST go to hazmat zones (YB19-20)

"""
        return [TextBlock(text=prompt)]

    # ----- Tools -----

    @tool
    async def observe_port(self, params: ObservePortParams) -> ToolOutput:
        """Get a full snapshot of berths, cranes, yard, gates, rail, weather, and disruptions."""
        if self.finished:
            return ToolOutput(
                blocks=[TextBlock(text="Simulation has ended.")],
                metadata={"error": "finished"}, reward=0.0, finished=True
            )
        status = self.sim.get_port_status()
        text = self._format_port_status(status)
        return ToolOutput(
            blocks=[TextBlock(text=text)],
            metadata=status, reward=0.0, finished=False
        )

    @tool
    async def assign_berth(self, params: AssignBerthParams) -> ToolOutput:
        """Dock a waiting vessel at a berth. Validates draft, availability, and tide."""
        if self.finished:
            return ToolOutput(
                blocks=[TextBlock(text="Simulation has ended.")],
                metadata={"error": "finished"}, reward=0.0, finished=True
            )
        result = self.sim.assign_berth(params.vessel_id, params.berth_id)
        if "error" in result:
            return ToolOutput(
                blocks=[TextBlock(text=f"Berth assignment failed: {result['error']}")],
                metadata=result, reward=0.0, finished=False
            )
        return ToolOutput(
            blocks=[TextBlock(text=result["message"])],
            metadata=result, reward=0.0, finished=False
        )

    @tool
    async def assign_cranes(self, params: AssignCranesParams) -> ToolOutput:
        """Assign idle cranes to a berthed vessel. Cranes must be at the same berth."""
        if self.finished:
            return ToolOutput(
                blocks=[TextBlock(text="Simulation has ended.")],
                metadata={"error": "finished"}, reward=0.0, finished=True
            )
        result = self.sim.assign_cranes(params.vessel_id, params.crane_ids)
        if "error" in result:
            return ToolOutput(
                blocks=[TextBlock(text=f"Crane assignment failed: {result['error']}")],
                metadata=result, reward=0.0, finished=False
            )
        return ToolOutput(
            blocks=[TextBlock(text=result["message"])],
            metadata=result, reward=0.0, finished=False
        )

    @tool
    async def move_crane(self, params: MoveCraneParams) -> ToolOutput:
        """Relocate a crane to a different berth (takes 1 hour). Blocked during storms."""
        if self.finished:
            return ToolOutput(
                blocks=[TextBlock(text="Simulation has ended.")],
                metadata={"error": "finished"}, reward=0.0, finished=True
            )
        result = self.sim.move_crane(params.crane_id, params.berth_id)
        if "error" in result:
            return ToolOutput(
                blocks=[TextBlock(text=f"Crane move failed: {result['error']}")],
                metadata=result, reward=0.0, finished=False
            )
        return ToolOutput(
            blocks=[TextBlock(text=result["message"])],
            metadata=result, reward=0.0, finished=False
        )

    @tool
    async def set_yard_plan(self, params: SetYardPlanParams) -> ToolOutput:
        """Designate yard blocks for a vessel's containers. Reefer needs power blocks, hazmat needs hazmat zones."""
        if self.finished:
            return ToolOutput(
                blocks=[TextBlock(text="Simulation has ended.")],
                metadata={"error": "finished"}, reward=0.0, finished=True
            )
        result = self.sim.set_yard_plan(params.vessel_id, params.yard_block_ids, params.container_type)
        if "error" in result:
            return ToolOutput(
                blocks=[TextBlock(text=f"Yard plan failed: {result['error']}")],
                metadata=result, reward=0.0, finished=False
            )
        return ToolOutput(
            blocks=[TextBlock(text=result["message"])],
            metadata=result, reward=0.0, finished=False
        )

    @tool
    async def dispatch_trucks(self, params: DispatchTrucksParams) -> ToolOutput:
        """Route trucks between a yard block and a gate lane."""
        if self.finished:
            return ToolOutput(
                blocks=[TextBlock(text="Simulation has ended.")],
                metadata={"error": "finished"}, reward=0.0, finished=True
            )
        result = self.sim.dispatch_trucks(params.count, params.yard_block_id, params.gate_id)
        if "error" in result:
            return ToolOutput(
                blocks=[TextBlock(text=f"Truck dispatch failed: {result['error']}")],
                metadata=result, reward=0.0, finished=False
            )
        return ToolOutput(
            blocks=[TextBlock(text=result["message"])],
            metadata=result, reward=0.0, finished=False
        )

    @tool
    async def schedule_train(self, params: ScheduleTrainParams) -> ToolOutput:
        """Schedule a rail departure from designated yard blocks."""
        if self.finished:
            return ToolOutput(
                blocks=[TextBlock(text="Simulation has ended.")],
                metadata={"error": "finished"}, reward=0.0, finished=True
            )
        result = self.sim.schedule_train(params.track_id, params.yard_block_ids, params.departure_hour)
        if "error" in result:
            return ToolOutput(
                blocks=[TextBlock(text=f"Train scheduling failed: {result['error']}")],
                metadata=result, reward=0.0, finished=False
            )
        return ToolOutput(
            blocks=[TextBlock(text=result["message"])],
            metadata=result, reward=0.0, finished=False
        )

    @tool
    async def advance_time(self, params: AdvanceTimeParams) -> ToolOutput:
        """Advance simulation by 1-12 hours, processing all events and operations. Returns per-step reward."""
        if self.finished:
            return ToolOutput(
                blocks=[TextBlock(text="Simulation has ended.")],
                metadata={"error": "finished"}, reward=0.0, finished=True
            )
        if params.hours < 1 or params.hours > 12:
            return ToolOutput(
                blocks=[TextBlock(text=f"Hours must be 1-12 (got {params.hours}).")],
                metadata={"error": "invalid_hours"}, reward=0.0, finished=False
            )

        target = self.sim.clock + params.hours
        events = self.sim.advance_to(target)
        step_reward = self.sim.compute_step_reward()
        self.sim.step_rewards.append(step_reward)

        if self.sim.clock >= self.sim.planning_horizon:
            final = self.sim.compute_final_reward()
            self.finished = True
            text = self._format_advance_events(events)
            text += "\n\n" + self._format_step_reward(step_reward)
            text += "\n\n" + self._format_final_result(final)
            final["events"] = events
            final["step_reward"] = step_reward
            return ToolOutput(
                blocks=[TextBlock(text=text)],
                metadata=final,
                reward=final["total_reward"],
                finished=True,
            )

        text = self._format_advance_events(events)
        text += "\n\n" + self._format_step_reward(step_reward)
        text += f"\n\nClock: {self.sim.clock:.1f}hr / {self.sim.planning_horizon:.0f}hr"

        return ToolOutput(
            blocks=[TextBlock(text=text)],
            metadata={"events": events, "step_reward": step_reward, "clock": self.sim.clock},
            reward=step_reward["weighted_total"],
            finished=False,
        )

    @tool
    async def handle_disruption(self, params: HandleDisruptionParams) -> ToolOutput:
        """Respond to an active disruption. Actions: accept, overtime, reroute, delay."""
        if self.finished:
            return ToolOutput(
                blocks=[TextBlock(text="Simulation has ended.")],
                metadata={"error": "finished"}, reward=0.0, finished=True
            )
        result = self.sim.handle_disruption(params.disruption_id, params.action)
        if "error" in result:
            return ToolOutput(
                blocks=[TextBlock(text=f"Disruption handling failed: {result['error']}")],
                metadata=result, reward=0.0, finished=False
            )
        return ToolOutput(
            blocks=[TextBlock(text=result["message"])],
            metadata=result, reward=0.0, finished=False
        )

    @tool
    async def submit_plan(self, params: SubmitPlanParams) -> ToolOutput:
        """End the simulation and compute the final reward."""
        if self.finished:
            return ToolOutput(
                blocks=[TextBlock(text="Simulation already ended.")],
                metadata={"error": "finished"}, reward=0.0, finished=True
            )
        # Compute one more step reward if needed
        if not self.sim.step_rewards:
            step = self.sim.compute_step_reward()
            self.sim.step_rewards.append(step)

        final = self.sim.compute_final_reward()
        self.finished = True
        text = self._format_final_result(final)
        return ToolOutput(
            blocks=[TextBlock(text=text)],
            metadata=final,
            reward=final["total_reward"],
            finished=True,
        )

    # ----- Formatting helpers -----

    def _format_port_status(self, status: dict) -> str:
        lines = [f"=== PORT STATUS at hour {status['clock']:.1f} / {status['planning_horizon']:.0f} ==="]
        lines.append(f"Wind: {status['wind_speed_knots']:.0f} knots | "
                    f"Tide: {'HIGH' if status['tide_high'] else 'LOW'} | "
                    f"Vessels: {status['vessels_departed']}/{status['vessels_total']} departed")

        # Berths
        lines.append("\n--- BERTHS ---")
        for b in status["berths"]:
            state = f"OCCUPIED by {b['vessel_id']} ({b.get('occupied_hours', 0):.1f}h)" if b["occupied"] else "EMPTY"
            lines.append(f"  {b['berth_id']:4s} | {b['length_m']:.0f}m, depth {b['draft_m']:.0f}m, "
                        f"max {b['max_cranes']} cranes | {state}")

        # Cranes
        lines.append("\n--- CRANES ---")
        for c in status["cranes"]:
            line = f"  {c['crane_id']:4s} | berth {c['berth_id']} | {c['status']:10s}"
            if c.get("vessel_id"):
                line += f" | working {c['vessel_id']}"
            if c.get("moving_to"):
                line += f" | moving to {c['moving_to']} (arr {c['arrives_at']:.1f}h)"
            if c.get("repair_done_at"):
                line += f" | REPAIR done {c['repair_done_at']:.1f}h"
            line += f" | total moves: {c['total_moves']}"
            lines.append(line)

        # Vessels
        lines.append("\n--- VESSELS ---")
        for v in status["vessels"]:
            line = f"  {v['vessel_id']:5s} | {v['type']:12s} | {v['teu_capacity']:5d} TEU | {v['status']:10s}"
            if v["status"] == "berthed":
                line += (f" | berth={v.get('berth_id', '?')}, "
                        f"cranes={len(v.get('cranes_assigned', []))}, "
                        f"remaining={v['remaining_moves']}")
            elif v["status"] == "waiting":
                line += f" | draft={v['draft_required_m']}m"
            elif v["status"] == "scheduled":
                line += f" | arrives ~hr {v.get('actual_arrival', v['scheduled_arrival']):.1f}"
            if v.get("departure_time"):
                line += f" | departed hr {v['departure_time']:.1f}"
            lines.append(line)

        # Yard summary
        ys = status["yard_summary"]
        lines.append(f"\n--- YARD --- ({ys['utilization_pct']}% utilized, "
                    f"{ys['total_occupancy']}/{ys['total_capacity']} TEU)")
        for yb in status["yard_blocks"]:
            tags = []
            if yb["has_power"]:
                tags.append("POWER")
            if yb["hazmat_zone"]:
                tags.append("HAZMAT")
            if yb["customs_held"] > 0:
                tags.append(f"HELD:{yb['customs_held']}")
            tag_str = f" [{','.join(tags)}]" if tags else ""
            lines.append(f"  {yb['block_id']:5s} | {yb['occupancy']:3d}/{yb['effective_capacity']} "
                        f"({yb['utilization_pct']:5.1f}%){tag_str}")

        # Gates
        lines.append("\n--- GATES ---")
        for g in status["gate_lanes"]:
            lines.append(f"  {g['lane_id']:5s} | {g['direction']:8s} | queue: {g['queue']:3d} | "
                        f"total processed: {g['total_processed']}")

        # Rail
        lines.append("\n--- RAIL ---")
        for r in status["rail_tracks"]:
            state = "DEPARTED" if r["departed"] else \
                    (f"scheduled hr {r['scheduled_departure']:.1f}" if r["scheduled_departure"] else "available")
            lines.append(f"  {r['track_id']:5s} | {r['current_load_teu']:3d}/{r['max_teu']} TEU | {state}")

        # Active disruptions
        active = [d for d in status["disruptions"] if d["active"]]
        if active:
            lines.append("\n--- ACTIVE DISRUPTIONS ---")
            for d in active:
                lines.append(f"  {d['disruption_id']}: {d['type']}, "
                            f"severity {d['severity']:.0%}, "
                            f"ends hr {d['end_hour']:.1f}"
                            + (f", action: {d['agent_action']}" if d['agent_action'] else ""))

        # Upcoming vessels
        if status["upcoming_vessels"]:
            lines.append("\n--- UPCOMING ARRIVALS ---")
            for uv in status["upcoming_vessels"]:
                lines.append(f"  {uv['vessel_id']}: {uv['type']}, {uv['teu']} TEU, "
                            f"arrives ~hr {uv['actual']:.1f}, draft {uv['draft']}m")

        return "\n".join(lines)

    def _format_advance_events(self, events: list) -> str:
        if not events:
            return "No events during this period."
        lines = [f"Events during advance ({len(events)} total):"]
        for ev in events:
            lines.append(f"  [{ev['time']:.1f}hr] {ev.get('message', ev['event_type'])}")
        return "\n".join(lines)

    def _format_step_reward(self, reward: dict) -> str:
        lines = ["Step Reward Breakdown:"]
        for key, weight in REWARD_WEIGHTS.items():
            val = reward.get(key, 0.0)
            lines.append(f"  {key:25s}: {val:.4f} (weight {weight:.0%})")
        lines.append(f"  {'WEIGHTED TOTAL':25s}: {reward['weighted_total']:.4f}")
        return "\n".join(lines)

    def _format_final_result(self, reward: dict) -> str:
        lines = [
            "=" * 55,
            "SIMULATION COMPLETE - FINAL RESULTS",
            "=" * 55,
            f"Total Reward: {reward['total_reward']:.4f}",
            "",
            f"  Vessels completed: {reward['vessels_completed']}/{reward['vessels_total']}",
            f"  Total crane moves: {reward['total_crane_moves']}",
            f"  Trains departed:   {reward['trains_departed']}",
            f"  Safety violations: {reward['safety_violations']}",
            f"  Simulation ended:  hour {reward['clock']:.1f}",
            "",
            "Component Averages:",
        ]
        for key in REWARD_WEIGHTS:
            avg_key = f"avg_{key}"
            val = reward.get(avg_key, 0.0)
            lines.append(f"  {key:25s}: {val:.4f}")
        lines.append("=" * 55)
        return "\n".join(lines)
