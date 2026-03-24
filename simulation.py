"""
Discrete-event simulation engine for a container port terminal.

Models a medium-large container terminal with 4 berths, 12 STS cranes,
20 yard blocks, 8 gate lanes, and 4 rail tracks. Supports vessel scheduling,
crane assignment, yard management, truck/rail operations, and disruption handling.

Uses a hybrid approach: heapq event heap for discrete events (arrivals, breakdowns,
tides) combined with hourly processing for continuous operations (crane moves,
gate throughput).
"""

import heapq
import math
import random
from typing import Any, Dict, List, Optional, Tuple

from models import (
    Berth,
    Crane,
    CraneStatus,
    ContainerType,
    Disruption,
    DisruptionType,
    GateDirection,
    GateLane,
    PortEvent,
    RailTrack,
    Vessel,
    VesselStatus,
    VesselType,
    YardBlock,
    BERTH_CONFIGS,
    CRANE_BASELINE_MPH,
    CRANE_DISTRIBUTION,
    CRANE_EFFICIENCY_NOISE,
    CRANE_MOVE_DURATION,
    CRANE_MTBF_HOURS,
    CRANE_REPAIR_RANGE,
    DEFAULT_VESSEL_MIX,
    HAZMAT_BLOCKS,
    NUM_GATE_LANES_IN,
    NUM_GATE_LANES_OUT,
    NUM_RAIL_TRACKS,
    NUM_YARD_BLOCKS,
    PLANNING_HORIZON,
    REEFER_POWER_BLOCKS,
    REWARD_WEIGHTS,
    STORM_GATE_THROUGHPUT_FACTOR,
    VESSEL_CONFIGS,
    WIND_CRANE_HALT_KNOTS,
    WIND_GATE_REDUCTION_KNOTS,
)


class PortSimulation:
    """Core simulation engine for the container port terminal."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.rng = random.Random(config.get("seed", 42))
        self.clock: float = 0.0
        self.planning_horizon: float = PLANNING_HORIZON
        self._event_seq: int = 0
        self._event_heap: List[PortEvent] = []

        # Infrastructure
        self.berths: Dict[str, Berth] = {}
        self.cranes: Dict[str, Crane] = {}
        self.yard_blocks: Dict[str, YardBlock] = {}
        self.gate_lanes: Dict[str, GateLane] = {}
        self.rail_tracks: Dict[str, RailTrack] = {}

        # Vessels
        self.vessels: Dict[str, Vessel] = {}

        # Disruptions
        self.disruptions: Dict[str, Disruption] = {}
        self.active_disruptions: List[str] = []

        # Weather / tide state
        self.wind_speed_knots: float = 15.0
        self.tide_high: bool = False
        self.tide_windows: List[Tuple[float, float]] = []

        # Tracking for rewards
        self.step_rewards: List[Dict[str, float]] = []
        self.hourly_crane_moves: List[int] = []
        self.hourly_crane_working: List[int] = []
        self.truck_wait_times: List[float] = []
        self.safety_violations: int = 0
        self.overtime_penalty: float = 0.0
        self.departed_vessels: List[str] = []
        self.trains_departed: List[Dict[str, Any]] = []
        self._last_processed_hour: float = -1.0

        # Initialize
        self._init_infrastructure()
        self._init_vessels()
        self._init_disruptions()
        self._init_tide_schedule()
        self._schedule_vessel_arrivals()

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _init_infrastructure(self):
        # Berths
        for bcfg in BERTH_CONFIGS:
            self.berths[bcfg["berth_id"]] = Berth(**bcfg)

        # Cranes
        for berth_id, crane_ids in CRANE_DISTRIBUTION.items():
            for cid in crane_ids:
                c = Crane(crane_id=cid, berth_id=berth_id)
                c.next_breakdown_hour = self._sample_breakdown_time(0.0)
                self.cranes[cid] = c

        # Yard blocks
        for i in range(1, NUM_YARD_BLOCKS + 1):
            bid = f"YB{i:02d}"
            yb = YardBlock(
                block_id=bid,
                has_power_points=(bid in REEFER_POWER_BLOCKS),
                hazmat_zone=(bid in HAZMAT_BLOCKS),
            )
            # Initialize occupancy from config
            init_occ = self.config.get("yard_initial_occupancy", 0.40)
            yb.current_occupancy = int(yb.effective_capacity * init_occ * self.rng.uniform(0.7, 1.3))
            yb.current_occupancy = max(0, min(yb.current_occupancy, yb.effective_capacity))
            yb.containers_by_type["dry"] = yb.current_occupancy
            self.yard_blocks[bid] = yb

        # Gate lanes
        for i in range(1, NUM_GATE_LANES_IN + 1):
            self.gate_lanes[f"GI{i:02d}"] = GateLane(
                lane_id=f"GI{i:02d}", direction=GateDirection.INBOUND
            )
        for i in range(1, NUM_GATE_LANES_OUT + 1):
            self.gate_lanes[f"GO{i:02d}"] = GateLane(
                lane_id=f"GO{i:02d}", direction=GateDirection.OUTBOUND
            )

        # Rail tracks
        for i in range(1, NUM_RAIL_TRACKS + 1):
            self.rail_tracks[f"RT{i:02d}"] = RailTrack(track_id=f"RT{i:02d}")

    def _init_vessels(self):
        num_vessels = self.config.get("num_vessels", 9)
        vessel_mix_raw = self.config.get("vessel_mix", {
            "feeder": 0.30, "panamax": 0.35, "post_panamax": 0.25, "ulcv": 0.10,
        })
        reefer_frac = self.config.get("reefer_fraction", 0.05)
        hazmat_frac = self.config.get("hazmat_fraction", 0.02)
        delay_factor = self.config.get("vessel_delay_factor", 0.3)

        # Map string keys to VesselType
        vessel_mix = {}
        for k, v in vessel_mix_raw.items():
            vtype = VesselType(k) if isinstance(k, str) else k
            vessel_mix[vtype] = v

        # Generate vessel types based on mix
        types = []
        for vtype, fraction in vessel_mix.items():
            count = max(1, round(num_vessels * fraction))
            types.extend([vtype] * count)
        # Trim or extend to exact count
        self.rng.shuffle(types)
        types = types[:num_vessels]
        while len(types) < num_vessels:
            types.append(self.rng.choice(list(vessel_mix.keys())))

        # Spread arrivals across the planning horizon
        arrival_spacing = self.planning_horizon / (num_vessels + 1)

        for i, vtype in enumerate(types):
            vcfg = VESSEL_CONFIGS[vtype]
            teu = self.rng.randint(*vcfg["teu_range"])
            exchange_frac = self.rng.uniform(*vcfg["exchange_fraction"])
            total_exchange = int(teu * exchange_frac)
            import_ct = total_exchange // 2
            export_ct = total_exchange - import_ct

            reefer_ct = int(total_exchange * reefer_frac)
            hazmat_ct = int(total_exchange * hazmat_frac)

            draft = round(self.rng.uniform(*vcfg["draft_range"]), 1)
            target_ta = self.rng.uniform(*vcfg["target_turnaround"])

            # Scheduled arrival with some jitter
            base_arrival = arrival_spacing * (i + 1)
            jitter = self.rng.uniform(-arrival_spacing * 0.3, arrival_spacing * 0.3)
            scheduled = max(0.5, base_arrival + jitter)

            # Apply delay factor: some vessels arrive late
            if self.rng.random() < delay_factor:
                delay_hours = self.rng.uniform(2.0, 24.0 * delay_factor)
                actual = scheduled + delay_hours
            else:
                actual = scheduled + self.rng.uniform(-1.0, 1.0)
            actual = max(0.5, actual)

            vessel = Vessel(
                vessel_id=f"V{i+1:03d}",
                vessel_type=vtype,
                teu_capacity=teu,
                import_containers=import_ct,
                export_containers=export_ct,
                reefer_count=reefer_ct,
                hazmat_count=hazmat_ct,
                draft_required_m=draft,
                min_cranes=vcfg["min_cranes"],
                max_cranes=vcfg["max_cranes"],
                target_turnaround_hours=round(target_ta, 1),
                scheduled_arrival=round(scheduled, 1),
                actual_arrival=round(actual, 1),
                priority=self.rng.randint(1, 5),
            )
            self.vessels[vessel.vessel_id] = vessel

    def _init_disruptions(self):
        disruption_specs = self.config.get("disruptions", [])
        for i, spec in enumerate(disruption_specs):
            dtype = DisruptionType(spec["type"])
            did = f"D{i+1:03d}_{dtype.value}"
            self.disruptions[did] = Disruption(
                disruption_id=did,
                disruption_type=dtype,
                start_hour=spec["start_hour"],
                end_hour=spec["end_hour"],
                severity=spec.get("severity", 0.5),
                details=spec.get("details", {}),
            )
            self._push_event(spec["start_hour"], "disruption_start", did)
            self._push_event(spec["end_hour"], "disruption_end", did)

    def _init_tide_schedule(self):
        """Generate semi-diurnal tide windows: 2 high tides per day, ~4h each."""
        # First high tide starts at a random offset 0-6 hours
        offset = self.rng.uniform(0.0, 6.0)
        period = 12.42  # hours between high tides (semi-diurnal)
        window_duration = 4.0

        t = offset
        while t < self.planning_horizon + 24:
            start = t
            end = t + window_duration
            self.tide_windows.append((round(start, 2), round(end, 2)))
            self._push_event(start, "tide_high_start", None)
            self._push_event(end, "tide_high_end", None)
            t += period

        # Check initial tide
        for start, end in self.tide_windows:
            if start <= 0.0 < end:
                self.tide_high = True
                break

    def _schedule_vessel_arrivals(self):
        for vessel in self.vessels.values():
            if vessel.actual_arrival is not None:
                self._push_event(vessel.actual_arrival, "vessel_arrival", vessel.vessel_id)

    def _push_event(self, time: float, event_type: str, entity_id: Optional[str] = None,
                    details: Optional[Dict[str, Any]] = None):
        self._event_seq += 1
        ev = PortEvent(time=time, sequence=self._event_seq, event_type=event_type,
                       entity_id=entity_id, details=details)
        heapq.heappush(self._event_heap, ev)

    def _sample_breakdown_time(self, from_hour: float) -> float:
        """Sample next breakdown time using exponential inter-arrival."""
        mtbf = CRANE_MTBF_HOURS
        # For equipment_aging scenarios, halve MTBF
        if self.config.get("scenario_type") == "equipment_aging":
            mtbf = mtbf / 2
        return from_hour + self.rng.expovariate(1.0 / mtbf)

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def advance_to(self, target_time: float) -> List[Dict[str, Any]]:
        """Advance simulation to target_time, processing events and hourly ops."""
        target_time = min(target_time, self.planning_horizon)
        events_log: List[Dict[str, Any]] = []

        while self._event_heap and self._event_heap[0].time <= target_time:
            ev = heapq.heappop(self._event_heap)
            self.clock = ev.time
            result = self._process_event(ev)
            if result:
                events_log.append(result)

        # Process hourly operations for each hour we pass through
        start_hour = int(self._last_processed_hour) + 1
        end_hour = int(target_time)
        for hour in range(max(start_hour, 0), end_hour + 1):
            h = float(hour)
            if h <= self._last_processed_hour:
                continue
            if h > target_time:
                break
            self._process_hourly_operations(h)
            self._last_processed_hour = h

        self.clock = target_time
        return events_log

    def _process_event(self, event: PortEvent) -> Optional[Dict[str, Any]]:
        handlers = {
            "vessel_arrival": self._handle_vessel_arrival,
            "crane_move_complete": self._handle_crane_move_complete,
            "crane_breakdown": self._handle_crane_breakdown,
            "crane_repair": self._handle_crane_repair,
            "disruption_start": self._handle_disruption_start,
            "disruption_end": self._handle_disruption_end,
            "tide_high_start": self._handle_tide_high_start,
            "tide_high_end": self._handle_tide_high_end,
            "vessel_departure": self._handle_vessel_departure,
            "train_departure": self._handle_train_departure,
        }
        handler = handlers.get(event.event_type)
        if handler:
            return handler(event)
        return None

    def _handle_vessel_arrival(self, event: PortEvent) -> Dict[str, Any]:
        vessel = self.vessels.get(event.entity_id)
        if not vessel:
            return {"time": event.time, "event_type": "vessel_arrival",
                    "message": f"Unknown vessel {event.entity_id}"}
        vessel.status = VesselStatus.WAITING
        delay = vessel.actual_arrival - vessel.scheduled_arrival
        delay_str = f" ({delay:+.1f}h vs schedule)" if abs(delay) > 0.5 else " (on schedule)"
        return {
            "time": event.time, "event_type": "vessel_arrival",
            "vessel_id": vessel.vessel_id,
            "message": f"Vessel {vessel.vessel_id} ({vessel.vessel_type.value}, "
                       f"{vessel.teu_capacity} TEU) arrived{delay_str}. "
                       f"Draft: {vessel.draft_required_m}m, "
                       f"Moves: {vessel.total_moves} ({vessel.import_containers} import, "
                       f"{vessel.export_containers} export)",
        }

    def _handle_vessel_departure(self, event: PortEvent) -> Dict[str, Any]:
        vessel = self.vessels.get(event.entity_id)
        if not vessel:
            return {"time": event.time, "event_type": "vessel_departure",
                    "message": f"Unknown vessel {event.entity_id}"}
        vessel.status = VesselStatus.DEPARTED
        vessel.departure_time = event.time
        self.departed_vessels.append(vessel.vessel_id)

        # Free berth
        berth = self.berths.get(vessel.berth_id)
        if berth:
            berth.vessel_id = None
            berth.occupied_since = None

        # Release cranes
        for cid in vessel.cranes_assigned:
            crane = self.cranes.get(cid)
            if crane:
                crane.status = CraneStatus.IDLE
                crane.vessel_id = None
        vessel.cranes_assigned = []

        turnaround = event.time - (vessel.berthing_time or vessel.actual_arrival)
        return {
            "time": event.time, "event_type": "vessel_departure",
            "vessel_id": vessel.vessel_id,
            "message": f"Vessel {vessel.vessel_id} departed. "
                       f"Turnaround: {turnaround:.1f}h "
                       f"(target: {vessel.target_turnaround_hours:.1f}h). "
                       f"Unloaded: {vessel.containers_unloaded}/{vessel.import_containers}, "
                       f"Loaded: {vessel.containers_loaded}/{vessel.export_containers}",
        }

    def _handle_crane_move_complete(self, event: PortEvent) -> Dict[str, Any]:
        crane = self.cranes.get(event.entity_id)
        if not crane:
            return {"time": event.time, "event_type": "crane_move_complete",
                    "message": f"Unknown crane {event.entity_id}"}
        crane.berth_id = crane.target_berth_id or crane.berth_id
        crane.target_berth_id = None
        crane.move_end_hour = None
        crane.status = CraneStatus.IDLE
        return {
            "time": event.time, "event_type": "crane_move_complete",
            "crane_id": crane.crane_id,
            "message": f"Crane {crane.crane_id} relocated to berth {crane.berth_id}",
        }

    def _handle_crane_breakdown(self, event: PortEvent) -> Dict[str, Any]:
        crane = self.cranes.get(event.entity_id)
        if not crane or crane.status == CraneStatus.BREAKDOWN:
            return None
        old_status = crane.status
        crane.status = CraneStatus.BREAKDOWN
        # Remove from vessel assignment if working
        if crane.vessel_id:
            vessel = self.vessels.get(crane.vessel_id)
            if vessel and crane.crane_id in vessel.cranes_assigned:
                vessel.cranes_assigned.remove(crane.crane_id)
            crane.vessel_id = None

        repair_hours = self.rng.uniform(*CRANE_REPAIR_RANGE)
        crane.repair_end_hour = event.time + repair_hours
        self._push_event(crane.repair_end_hour, "crane_repair", crane.crane_id)

        return {
            "time": event.time, "event_type": "crane_breakdown",
            "crane_id": crane.crane_id,
            "message": f"Crane {crane.crane_id} broke down! "
                       f"Repair estimated: {repair_hours:.1f}h "
                       f"(back online at hour {crane.repair_end_hour:.1f})",
        }

    def _handle_crane_repair(self, event: PortEvent) -> Dict[str, Any]:
        crane = self.cranes.get(event.entity_id)
        if not crane or crane.status != CraneStatus.BREAKDOWN:
            return None
        crane.status = CraneStatus.IDLE
        crane.repair_end_hour = None
        # Schedule next breakdown
        crane.next_breakdown_hour = self._sample_breakdown_time(event.time)
        self._push_event(crane.next_breakdown_hour, "crane_breakdown", crane.crane_id)
        return {
            "time": event.time, "event_type": "crane_repair",
            "crane_id": crane.crane_id,
            "message": f"Crane {crane.crane_id} repaired and back online at berth {crane.berth_id}",
        }

    def _handle_disruption_start(self, event: PortEvent) -> Dict[str, Any]:
        d = self.disruptions.get(event.entity_id)
        if not d:
            return None
        d.active = True
        self.active_disruptions.append(d.disruption_id)

        if d.disruption_type == DisruptionType.STORM:
            self.wind_speed_knots = d.details.get("wind_knots", 50.0)
        elif d.disruption_type == DisruptionType.EQUIPMENT_BREAKDOWN:
            crane_id = d.details.get("crane_id")
            if crane_id and crane_id in self.cranes:
                self._push_event(event.time, "crane_breakdown", crane_id)

        return {
            "time": event.time, "event_type": "disruption_start",
            "disruption_id": d.disruption_id,
            "message": f"DISRUPTION: {d.disruption_type.value} started! "
                       f"Severity: {d.severity:.0%}. "
                       f"Expected end: hour {d.end_hour:.1f}. "
                       f"Details: {d.details}",
        }

    def _handle_disruption_end(self, event: PortEvent) -> Dict[str, Any]:
        d = self.disruptions.get(event.entity_id)
        if not d:
            return None
        d.active = False
        d.resolved = True
        if d.disruption_id in self.active_disruptions:
            self.active_disruptions.remove(d.disruption_id)

        if d.disruption_type == DisruptionType.STORM:
            # Reset wind unless another storm is active
            still_storming = any(
                self.disruptions[did].disruption_type == DisruptionType.STORM
                and self.disruptions[did].active
                for did in self.active_disruptions
            )
            if not still_storming:
                self.wind_speed_knots = 15.0

        return {
            "time": event.time, "event_type": "disruption_end",
            "disruption_id": d.disruption_id,
            "message": f"Disruption {d.disruption_type.value} ended.",
        }

    def _handle_tide_high_start(self, event: PortEvent) -> Dict[str, Any]:
        self.tide_high = True
        return {
            "time": event.time, "event_type": "tide_high_start",
            "message": "High tide window opened. Deep-draft vessels may now berth/depart.",
        }

    def _handle_tide_high_end(self, event: PortEvent) -> Dict[str, Any]:
        self.tide_high = False
        return {
            "time": event.time, "event_type": "tide_high_end",
            "message": "High tide window closed. Deep-draft vessels cannot berth/depart.",
        }

    def _handle_train_departure(self, event: PortEvent) -> Dict[str, Any]:
        track = self.rail_tracks.get(event.entity_id)
        if not track or track.departed:
            return None

        # Load containers from designated blocks
        total_loaded = 0
        for block_id in track.loading_from_blocks:
            block = self.yard_blocks.get(block_id)
            if block and block.current_occupancy > 0:
                # Load up to remaining capacity
                can_load = min(
                    block.current_occupancy,
                    track.max_teu - track.current_load_teu - total_loaded
                )
                if can_load > 0:
                    block.current_occupancy -= can_load
                    total_loaded += can_load
                    # Reduce type counts proportionally
                    remaining_reduce = can_load
                    for ctype in list(block.containers_by_type.keys()):
                        reduce = min(block.containers_by_type.get(ctype, 0), remaining_reduce)
                        block.containers_by_type[ctype] = block.containers_by_type.get(ctype, 0) - reduce
                        remaining_reduce -= reduce
                        if remaining_reduce <= 0:
                            break

        track.current_load_teu += total_loaded
        track.departed = True
        track.departure_actual = event.time

        fill_rate = track.current_load_teu / max(1, track.max_teu)
        self.trains_departed.append({
            "track_id": track.track_id,
            "teu_loaded": track.current_load_teu,
            "max_teu": track.max_teu,
            "fill_rate": fill_rate,
            "departure_hour": event.time,
        })

        return {
            "time": event.time, "event_type": "train_departure",
            "track_id": track.track_id,
            "message": f"Train on {track.track_id} departed with "
                       f"{track.current_load_teu}/{track.max_teu} TEU "
                       f"(fill rate: {fill_rate:.0%})",
        }

    # ------------------------------------------------------------------
    # Hourly processing
    # ------------------------------------------------------------------

    def _process_hourly_operations(self, hour: float):
        self._process_crane_operations(hour)
        self._process_gate_operations(hour)
        self._check_crane_breakdowns(hour)
        self._process_customs_holds(hour)

    def _get_storm_factor(self) -> float:
        """Return crane efficiency factor based on wind. 0 if above halt threshold."""
        if self.wind_speed_knots >= WIND_CRANE_HALT_KNOTS:
            return 0.0
        if self.wind_speed_knots >= WIND_GATE_REDUCTION_KNOTS:
            # Linear reduction between 30 and 40 knots
            return 1.0 - (self.wind_speed_knots - WIND_GATE_REDUCTION_KNOTS) / \
                   (WIND_CRANE_HALT_KNOTS - WIND_GATE_REDUCTION_KNOTS)
        return 1.0

    def _get_strike_factor(self) -> float:
        """Return productivity factor during labor strikes."""
        for did in self.active_disruptions:
            d = self.disruptions.get(did)
            if d and d.disruption_type == DisruptionType.LABOR_STRIKE and d.active:
                return d.details.get("productivity_factor", 0.5)
        return 1.0

    def _get_overtime_factor(self) -> float:
        """Return overtime boost (from agent handle_disruption action)."""
        for did in self.active_disruptions:
            d = self.disruptions.get(did)
            if d and d.active and d.agent_action == "overtime":
                return 1.5
        return 1.0

    def _process_crane_operations(self, hour: float):
        storm_factor = self._get_storm_factor()
        strike_factor = self._get_strike_factor()
        overtime_factor = self._get_overtime_factor()

        hour_moves = 0
        hour_working = 0

        for crane in self.cranes.values():
            crane.actual_moves_this_hour = 0
            if crane.status != CraneStatus.WORKING or not crane.vessel_id:
                continue

            vessel = self.vessels.get(crane.vessel_id)
            if not vessel or vessel.remaining_moves <= 0:
                continue

            hour_working += 1
            efficiency = storm_factor * strike_factor * overtime_factor
            noise = self.rng.uniform(*CRANE_EFFICIENCY_NOISE)
            moves = int(crane.moves_per_hour * efficiency * noise)
            moves = max(0, moves)

            # Apply moves to vessel
            remaining_import = vessel.import_containers - vessel.containers_unloaded
            remaining_export = vessel.export_containers - vessel.containers_loaded

            # Prioritize unloading, then loading
            unload_moves = min(moves, remaining_import)
            vessel.containers_unloaded += unload_moves
            load_moves = min(moves - unload_moves, remaining_export)
            vessel.containers_loaded += load_moves

            actual = unload_moves + load_moves
            crane.actual_moves_this_hour = actual
            crane.total_moves += actual
            crane.operational_hours += 1.0
            hour_moves += actual

            # Update yard occupancy for unloaded containers
            if unload_moves > 0:
                self._add_containers_to_yard(vessel, unload_moves)
            if load_moves > 0:
                self._remove_containers_from_yard(vessel, load_moves)

            # Check vessel completion
            if vessel.remaining_moves <= 0:
                self._initiate_vessel_departure(vessel, hour)

        self.hourly_crane_moves.append(hour_moves)
        self.hourly_crane_working.append(hour_working)

    def _add_containers_to_yard(self, vessel: Vessel, count: int):
        """Add unloaded containers to vessel's designated yard blocks."""
        blocks = vessel.yard_blocks_import
        if not blocks:
            # Default: spread across first available blocks
            blocks = [b.block_id for b in self.yard_blocks.values()
                      if b.current_occupancy < b.effective_capacity][:3]
            if not blocks:
                blocks = list(self.yard_blocks.keys())[:1]

        per_block = max(1, count // len(blocks))
        remaining = count
        for bid in blocks:
            block = self.yard_blocks.get(bid)
            if not block or remaining <= 0:
                break
            add = min(per_block, remaining, block.effective_capacity - block.current_occupancy)
            if add > 0:
                block.current_occupancy += add
                block.containers_by_vessel[vessel.vessel_id] = \
                    block.containers_by_vessel.get(vessel.vessel_id, 0) + add
                block.containers_by_type["dry"] = \
                    block.containers_by_type.get("dry", 0) + add
                remaining -= add

    def _remove_containers_from_yard(self, vessel: Vessel, count: int):
        """Remove export containers from yard blocks."""
        blocks = vessel.yard_blocks_export
        if not blocks:
            blocks = [b.block_id for b in self.yard_blocks.values()
                      if b.current_occupancy > 0][:3]

        remaining = count
        for bid in blocks:
            block = self.yard_blocks.get(bid)
            if not block or remaining <= 0:
                break
            remove = min(remaining, block.current_occupancy)
            if remove > 0:
                block.current_occupancy -= remove
                block.containers_by_type["dry"] = \
                    max(0, block.containers_by_type.get("dry", 0) - remove)
                remaining -= remove

    def _initiate_vessel_departure(self, vessel: Vessel, hour: float):
        """Start departure process. Deep-draft vessels wait for high tide."""
        if vessel.status == VesselStatus.DEPARTING or vessel.status == VesselStatus.DEPARTED:
            return
        vessel.status = VesselStatus.DEPARTING

        if vessel.is_deep_draft and not self.tide_high:
            # Find next high tide window
            for start, end in self.tide_windows:
                if start > hour:
                    self._push_event(start + 0.5, "vessel_departure", vessel.vessel_id)
                    return
            # Fallback: depart in 1 hour
            self._push_event(hour + 1.0, "vessel_departure", vessel.vessel_id)
        else:
            self._push_event(hour + 1.0, "vessel_departure", vessel.vessel_id)

    def _process_gate_operations(self, hour: float):
        gate_factor = 1.0
        if self.wind_speed_knots >= WIND_GATE_REDUCTION_KNOTS:
            gate_factor = STORM_GATE_THROUGHPUT_FACTOR

        for lane in self.gate_lanes.values():
            effective_throughput = int(lane.throughput_per_hour * gate_factor)
            processed = min(lane.current_queue, effective_throughput)
            lane.trucks_processed_this_hour = processed
            lane.total_trucks_processed += processed

            # Track wait times for queued trucks
            for _ in range(processed):
                wait = lane.current_queue / max(1, effective_throughput) * 60.0  # minutes
                self.truck_wait_times.append(wait)

            lane.current_queue = max(0, lane.current_queue - processed)

    def _check_crane_breakdowns(self, hour: float):
        """Check if any crane breakdowns should occur this hour."""
        for crane in self.cranes.values():
            if (crane.next_breakdown_hour is not None
                    and crane.next_breakdown_hour <= hour
                    and crane.status not in (CraneStatus.BREAKDOWN, CraneStatus.MOVING)):
                self._push_event(hour, "crane_breakdown", crane.crane_id)
                crane.next_breakdown_hour = None  # Will be rescheduled on repair

    def _process_customs_holds(self, hour: float):
        """Apply customs inspection holds to containers in yard."""
        inspection_rate = 0.03  # Default 3%
        for did in self.active_disruptions:
            d = self.disruptions.get(did)
            if d and d.disruption_type == DisruptionType.CUSTOMS_HOLD and d.active:
                inspection_rate = d.details.get("inspection_rate", 0.15)

        # Apply to random subset of newly arrived containers
        for block in self.yard_blocks.values():
            if block.current_occupancy > 0 and self.rng.random() < inspection_rate * 0.1:
                hold_count = max(1, int(block.current_occupancy * inspection_rate * 0.05))
                block.customs_held = min(
                    block.customs_held + hold_count,
                    block.current_occupancy
                )

    # ------------------------------------------------------------------
    # Agent action methods
    # ------------------------------------------------------------------

    def assign_berth(self, vessel_id: str, berth_id: str) -> Dict[str, Any]:
        vessel = self.vessels.get(vessel_id)
        if not vessel:
            return {"error": f"Vessel {vessel_id} not found"}
        if vessel.status != VesselStatus.WAITING:
            return {"error": f"Vessel {vessel_id} is {vessel.status.value}, not waiting"}

        berth = self.berths.get(berth_id)
        if not berth:
            return {"error": f"Berth {berth_id} not found"}
        if berth.vessel_id is not None:
            return {"error": f"Berth {berth_id} is occupied by {berth.vessel_id}"}
        if vessel.draft_required_m > berth.draft_m:
            return {"error": f"Vessel draft {vessel.draft_required_m}m exceeds "
                           f"berth depth {berth.draft_m}m"}

        # Tide check for deep-draft vessels
        if vessel.is_deep_draft and not self.tide_high:
            return {"error": f"Vessel {vessel_id} requires high tide to berth "
                           f"(draft {vessel.draft_required_m}m > 14.0m). "
                           f"Wait for next high tide window."}

        # Berth the vessel
        vessel.status = VesselStatus.BERTHED
        vessel.berth_id = berth_id
        vessel.berthing_time = self.clock
        berth.vessel_id = vessel_id
        berth.occupied_since = self.clock

        return {
            "message": f"Vessel {vessel_id} berthed at {berth_id}. "
                       f"Draft: {vessel.draft_required_m}m / {berth.draft_m}m. "
                       f"Max cranes: {berth.max_cranes}. "
                       f"Total moves needed: {vessel.total_moves}",
            "vessel_id": vessel_id,
            "berth_id": berth_id,
        }

    def assign_cranes(self, vessel_id: str, crane_ids: List[str]) -> Dict[str, Any]:
        vessel = self.vessels.get(vessel_id)
        if not vessel:
            return {"error": f"Vessel {vessel_id} not found"}
        if vessel.status != VesselStatus.BERTHED:
            return {"error": f"Vessel {vessel_id} is {vessel.status.value}, not berthed"}
        if not vessel.berth_id:
            return {"error": f"Vessel {vessel_id} has no berth assigned"}

        berth = self.berths.get(vessel.berth_id)
        if not berth:
            return {"error": f"Berth {vessel.berth_id} not found"}

        # Check total cranes (existing + new)
        new_crane_ids = [c for c in crane_ids if c not in vessel.cranes_assigned]
        total_after = len(vessel.cranes_assigned) + len(new_crane_ids)
        if total_after > berth.max_cranes:
            return {"error": f"Total cranes ({total_after}) exceeds berth max ({berth.max_cranes})"}
        if total_after > vessel.max_cranes:
            return {"error": f"Total cranes ({total_after}) exceeds vessel max ({vessel.max_cranes})"}

        # Validate each crane
        errors = []
        for cid in new_crane_ids:
            crane = self.cranes.get(cid)
            if not crane:
                errors.append(f"Crane {cid} not found")
                continue
            if crane.status != CraneStatus.IDLE:
                errors.append(f"Crane {cid} is {crane.status.value}")
                continue
            if crane.berth_id != vessel.berth_id:
                errors.append(f"Crane {cid} is at berth {crane.berth_id}, "
                             f"not {vessel.berth_id}")
                continue

        if errors:
            return {"error": "; ".join(errors)}

        # Assign cranes
        for cid in new_crane_ids:
            crane = self.cranes[cid]
            crane.status = CraneStatus.WORKING
            crane.vessel_id = vessel_id
            vessel.cranes_assigned.append(cid)

        return {
            "message": f"Assigned {len(new_crane_ids)} crane(s) to vessel {vessel_id}. "
                       f"Total cranes now: {len(vessel.cranes_assigned)}. "
                       f"Remaining moves: {vessel.remaining_moves}",
            "vessel_id": vessel_id,
            "crane_ids": vessel.cranes_assigned[:],
        }

    def move_crane(self, crane_id: str, berth_id: str) -> Dict[str, Any]:
        crane = self.cranes.get(crane_id)
        if not crane:
            return {"error": f"Crane {crane_id} not found"}
        if crane.status not in (CraneStatus.IDLE,):
            return {"error": f"Crane {crane_id} is {crane.status.value}, must be idle to move"}
        if crane.berth_id == berth_id:
            return {"error": f"Crane {crane_id} is already at berth {berth_id}"}

        berth = self.berths.get(berth_id)
        if not berth:
            return {"error": f"Berth {berth_id} not found"}

        # Storm check
        if self.wind_speed_knots >= WIND_CRANE_HALT_KNOTS:
            return {"error": f"Cannot move crane during storm (wind {self.wind_speed_knots:.0f} knots)"}

        crane.status = CraneStatus.MOVING
        crane.target_berth_id = berth_id
        crane.move_end_hour = self.clock + CRANE_MOVE_DURATION
        self._push_event(crane.move_end_hour, "crane_move_complete", crane_id)

        return {
            "message": f"Crane {crane_id} moving from {crane.berth_id} to {berth_id}. "
                       f"Arrives at hour {crane.move_end_hour:.1f}",
            "crane_id": crane_id,
            "from_berth": crane.berth_id,
            "to_berth": berth_id,
        }

    def set_yard_plan(self, vessel_id: str, block_ids: List[str],
                      container_type: str) -> Dict[str, Any]:
        vessel = self.vessels.get(vessel_id)
        if not vessel:
            return {"error": f"Vessel {vessel_id} not found"}
        if vessel.status not in (VesselStatus.WAITING, VesselStatus.BERTHED):
            return {"error": f"Vessel {vessel_id} is {vessel.status.value}"}

        ctype = container_type.lower()
        errors = []
        for bid in block_ids:
            block = self.yard_blocks.get(bid)
            if not block:
                errors.append(f"Block {bid} not found")
                continue
            if ctype == "reefer" and not block.has_power_points:
                errors.append(f"Block {bid} has no power points for reefer containers")
            if ctype == "hazmat" and not block.hazmat_zone:
                errors.append(f"Block {bid} is not a hazmat zone")
            if block.current_occupancy >= block.effective_capacity:
                errors.append(f"Block {bid} is at capacity ({block.current_occupancy}/{block.effective_capacity})")

        if errors:
            return {"error": "; ".join(errors)}

        # Set yard plan
        if ctype == "hazmat" or ctype == "reefer":
            # Special cargo uses separate allocation
            vessel.yard_blocks_import = list(set(vessel.yard_blocks_import + block_ids))
        else:
            vessel.yard_blocks_import = block_ids
        vessel.yard_blocks_export = block_ids  # Same blocks for simplicity

        return {
            "message": f"Yard plan set for vessel {vessel_id}: "
                       f"blocks {', '.join(block_ids)} for {ctype} containers. "
                       f"Available capacity: {sum(self.yard_blocks[b].effective_capacity - self.yard_blocks[b].current_occupancy for b in block_ids if b in self.yard_blocks)} TEU",
            "vessel_id": vessel_id,
            "block_ids": block_ids,
            "container_type": ctype,
        }

    def dispatch_trucks(self, count: int, yard_block_id: str,
                       gate_id: str) -> Dict[str, Any]:
        block = self.yard_blocks.get(yard_block_id)
        if not block:
            return {"error": f"Block {yard_block_id} not found"}

        gate = self.gate_lanes.get(gate_id)
        if not gate:
            return {"error": f"Gate {gate_id} not found"}

        if count < 1 or count > 100:
            return {"error": f"Truck count must be 1-100 (got {count})"}

        # Add trucks to gate queue
        gate.current_queue += count

        # Remove containers from yard (for outbound)
        if gate.direction == GateDirection.OUTBOUND:
            removed = min(count, block.current_occupancy)
            block.current_occupancy -= removed
            block.containers_by_type["dry"] = \
                max(0, block.containers_by_type.get("dry", 0) - removed)
        else:
            # Inbound: add containers to block
            added = min(count, block.effective_capacity - block.current_occupancy)
            block.current_occupancy += added
            block.containers_by_type["dry"] = \
                block.containers_by_type.get("dry", 0) + added

        return {
            "message": f"Dispatched {count} trucks: block {yard_block_id} <-> gate {gate_id}. "
                       f"Gate queue: {gate.current_queue}. "
                       f"Block occupancy: {block.current_occupancy}/{block.effective_capacity}",
            "count": count,
            "yard_block_id": yard_block_id,
            "gate_id": gate_id,
        }

    def schedule_train(self, track_id: str, block_ids: List[str],
                      departure_hour: float) -> Dict[str, Any]:
        track = self.rail_tracks.get(track_id)
        if not track:
            return {"error": f"Track {track_id} not found"}
        if track.departed:
            return {"error": f"Track {track_id} already departed"}
        if track.scheduled_departure is not None:
            return {"error": f"Track {track_id} already scheduled for hour {track.scheduled_departure:.1f}"}

        if departure_hour <= self.clock:
            return {"error": f"Departure hour {departure_hour:.1f} must be after current time {self.clock:.1f}"}

        errors = []
        for bid in block_ids:
            if bid not in self.yard_blocks:
                errors.append(f"Block {bid} not found")
        if errors:
            return {"error": "; ".join(errors)}

        track.scheduled_departure = departure_hour
        track.loading_from_blocks = block_ids

        # Estimate load for message (actual loading happens at departure)
        estimated_load = 0
        for bid in block_ids:
            block = self.yard_blocks.get(bid)
            if block and block.current_occupancy > 0:
                can_load = min(
                    block.current_occupancy,
                    track.max_teu - estimated_load
                )
                estimated_load += can_load

        self._push_event(departure_hour, "train_departure", track_id)

        return {
            "message": f"Train scheduled on {track_id} for hour {departure_hour:.1f}. "
                       f"Loading from blocks: {', '.join(block_ids)}. "
                       f"Estimated load: {estimated_load}/{track.max_teu} TEU",
            "track_id": track_id,
            "departure_hour": departure_hour,
            "estimated_load": estimated_load,
        }

    def handle_disruption(self, disruption_id: str, action: str) -> Dict[str, Any]:
        d = self.disruptions.get(disruption_id)
        if not d:
            return {"error": f"Disruption {disruption_id} not found"}
        if not d.active:
            return {"error": f"Disruption {disruption_id} is not active"}

        valid_actions = ["accept", "overtime", "reroute", "delay"]
        if action not in valid_actions:
            return {"error": f"Invalid action '{action}'. Must be one of: {valid_actions}"}

        d.agent_action = action
        msg = f"Disruption {disruption_id} ({d.disruption_type.value}): action '{action}' applied."

        if action == "overtime":
            self.overtime_penalty += 0.02  # 2% penalty to safety score per overtime
            msg += " Productivity +50% but overtime costs incurred."
        elif action == "reroute":
            msg += " Operations rerouted where possible."
        elif action == "delay":
            msg += " Operations delayed. Less risk but slower throughput."

        return {"message": msg, "disruption_id": disruption_id, "action": action}

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_port_status(self) -> Dict[str, Any]:
        berth_info = []
        for b in self.berths.values():
            bi = {
                "berth_id": b.berth_id, "length_m": b.length_m,
                "draft_m": b.draft_m, "max_cranes": b.max_cranes,
                "vessel_id": b.vessel_id, "occupied": b.vessel_id is not None,
            }
            if b.occupied_since is not None:
                bi["occupied_hours"] = round(self.clock - b.occupied_since, 1)
            berth_info.append(bi)

        crane_info = []
        for c in self.cranes.values():
            ci = {
                "crane_id": c.crane_id, "berth_id": c.berth_id,
                "status": c.status.value, "vessel_id": c.vessel_id,
                "total_moves": c.total_moves,
            }
            if c.move_end_hour:
                ci["moving_to"] = c.target_berth_id
                ci["arrives_at"] = c.move_end_hour
            if c.repair_end_hour:
                ci["repair_done_at"] = c.repair_end_hour
            crane_info.append(ci)

        yard_info = []
        for yb in self.yard_blocks.values():
            yard_info.append({
                "block_id": yb.block_id,
                "occupancy": yb.current_occupancy,
                "effective_capacity": yb.effective_capacity,
                "utilization_pct": round(100 * yb.current_occupancy / max(1, yb.effective_capacity), 1),
                "has_power": yb.has_power_points,
                "hazmat_zone": yb.hazmat_zone,
                "reefer_occupied": yb.reefer_occupied,
                "customs_held": yb.customs_held,
            })

        vessel_info = []
        for v in self.vessels.values():
            vi = {
                "vessel_id": v.vessel_id, "type": v.vessel_type.value,
                "teu_capacity": v.teu_capacity, "status": v.status.value,
                "import_containers": v.import_containers,
                "export_containers": v.export_containers,
                "reefer_count": v.reefer_count,
                "hazmat_count": v.hazmat_count,
                "draft_required_m": v.draft_required_m,
                "target_turnaround_hours": v.target_turnaround_hours,
                "scheduled_arrival": v.scheduled_arrival,
                "priority": v.priority,
            }
            if v.actual_arrival is not None:
                vi["actual_arrival"] = v.actual_arrival
            if v.berth_id:
                vi["berth_id"] = v.berth_id
            if v.cranes_assigned:
                vi["cranes_assigned"] = v.cranes_assigned[:]
            vi["containers_unloaded"] = v.containers_unloaded
            vi["containers_loaded"] = v.containers_loaded
            vi["remaining_moves"] = v.remaining_moves
            if v.berthing_time is not None:
                vi["berthing_time"] = v.berthing_time
                vi["hours_at_berth"] = round(self.clock - v.berthing_time, 1)
            if v.departure_time is not None:
                vi["departure_time"] = v.departure_time
            vessel_info.append(vi)

        gate_info = []
        for g in self.gate_lanes.values():
            gate_info.append({
                "lane_id": g.lane_id, "direction": g.direction.value,
                "queue": g.current_queue,
                "total_processed": g.total_trucks_processed,
            })

        rail_info = []
        for r in self.rail_tracks.values():
            rail_info.append({
                "track_id": r.track_id,
                "current_load_teu": r.current_load_teu,
                "max_teu": r.max_teu,
                "scheduled_departure": r.scheduled_departure,
                "departed": r.departed,
            })

        disruption_info = []
        for d in self.disruptions.values():
            disruption_info.append({
                "disruption_id": d.disruption_id,
                "type": d.disruption_type.value,
                "start_hour": d.start_hour,
                "end_hour": d.end_hour,
                "severity": d.severity,
                "active": d.active,
                "resolved": d.resolved,
                "agent_action": d.agent_action,
            })

        # Upcoming events
        upcoming_vessels = [
            {"vessel_id": v.vessel_id, "type": v.vessel_type.value,
             "scheduled": v.scheduled_arrival, "actual": v.actual_arrival,
             "teu": v.teu_capacity, "draft": v.draft_required_m}
            for v in self.vessels.values()
            if v.status == VesselStatus.SCHEDULED and v.actual_arrival is not None
               and v.actual_arrival > self.clock
        ]
        upcoming_vessels.sort(key=lambda x: x["actual"])

        # Compute aggregate stats
        total_yard_occ = sum(yb.current_occupancy for yb in self.yard_blocks.values())
        total_yard_cap = sum(yb.effective_capacity for yb in self.yard_blocks.values())

        return {
            "clock": round(self.clock, 1),
            "planning_horizon": self.planning_horizon,
            "hours_remaining": round(self.planning_horizon - self.clock, 1),
            "wind_speed_knots": self.wind_speed_knots,
            "tide_high": self.tide_high,
            "berths": berth_info,
            "cranes": crane_info,
            "yard_blocks": yard_info,
            "yard_summary": {
                "total_occupancy": total_yard_occ,
                "total_capacity": total_yard_cap,
                "utilization_pct": round(100 * total_yard_occ / max(1, total_yard_cap), 1),
            },
            "gate_lanes": gate_info,
            "rail_tracks": rail_info,
            "vessels": vessel_info,
            "disruptions": disruption_info,
            "upcoming_vessels": upcoming_vessels[:5],
            "vessels_departed": len(self.departed_vessels),
            "vessels_total": len(self.vessels),
        }

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------

    def compute_step_reward(self) -> Dict[str, float]:
        # 1. Berth utilization (0.20)
        occupied = sum(1 for b in self.berths.values() if b.vessel_id is not None)
        total_berths = len(self.berths)
        berth_util = occupied / max(1, total_berths)

        # 2. Crane productivity (0.20)
        working_cranes = [c for c in self.cranes.values()
                         if c.status == CraneStatus.WORKING]
        if working_cranes:
            actual_moves = sum(c.actual_moves_this_hour for c in working_cranes)
            benchmark = len(working_cranes) * CRANE_BASELINE_MPH
            crane_prod = min(1.0, actual_moves / max(1, benchmark))
        elif any(v.status in (VesselStatus.BERTHED, VesselStatus.WAITING)
                 for v in self.vessels.values()):
            crane_prod = 0.0  # Penalty: vessels waiting but no cranes working
        else:
            crane_prod = 0.5  # No vessels to work = neutral

        # 3. Vessel turnaround (0.20)
        turnaround_scores = []
        for v in self.vessels.values():
            if v.status == VesselStatus.DEPARTED and v.departure_time and v.berthing_time:
                actual = v.departure_time - v.berthing_time
                target = v.target_turnaround_hours
                ratio = actual / max(1, target)
                score = max(0.0, min(1.0, 2.0 - ratio))
                turnaround_scores.append(score)
            elif v.status == VesselStatus.BERTHED and v.berthing_time:
                elapsed = self.clock - v.berthing_time
                target = v.target_turnaround_hours
                if elapsed > target:
                    score = max(0.0, 1.0 - (elapsed - target) / max(1, target))
                    turnaround_scores.append(score)
                else:
                    turnaround_scores.append(1.0)
            elif v.status == VesselStatus.WAITING:
                # Penalty for waiting vessels
                if v.actual_arrival and self.clock > v.actual_arrival:
                    wait_hours = self.clock - v.actual_arrival
                    score = max(0.0, 1.0 - wait_hours / 24.0)
                    turnaround_scores.append(score)
        vessel_turn = (sum(turnaround_scores) / len(turnaround_scores)) if turnaround_scores else 0.3

        # 4. Yard efficiency (0.15)
        total_occ = sum(yb.current_occupancy for yb in self.yard_blocks.values())
        total_cap = sum(yb.effective_capacity for yb in self.yard_blocks.values())
        avg_util = total_occ / max(1, total_cap)
        if 0.50 <= avg_util <= 0.80:
            yard_eff = 1.0
        elif avg_util < 0.50:
            yard_eff = max(0.0, avg_util / 0.50)
        else:
            yard_eff = max(0.0, 1.0 - (avg_util - 0.80) / 0.20)

        # 5. Truck turnaround (0.10)
        recent_waits = self.truck_wait_times[-200:] if self.truck_wait_times else []
        if recent_waits:
            pct_under_60 = sum(1 for t in recent_waits if t <= 60.0) / len(recent_waits)
            truck_turn = pct_under_60
        else:
            truck_turn = 0.2  # low score for no truck activity

        # 6. Rail utilization (0.10)
        if self.trains_departed:
            rail_scores = [min(1.0, t["fill_rate"] / 0.80) for t in self.trains_departed]
            rail_util = sum(rail_scores) / len(rail_scores)
        else:
            rail_util = 0.1  # Low score for no rail activity

        # 7. Safety / compliance (0.05)
        violations = self._count_safety_violations()
        safety = max(0.0, 1.0 - violations * 0.2 - self.overtime_penalty)

        weighted = (
            REWARD_WEIGHTS["berth_utilization"] * berth_util +
            REWARD_WEIGHTS["crane_productivity"] * crane_prod +
            REWARD_WEIGHTS["vessel_turnaround"] * vessel_turn +
            REWARD_WEIGHTS["yard_efficiency"] * yard_eff +
            REWARD_WEIGHTS["truck_turnaround"] * truck_turn +
            REWARD_WEIGHTS["rail_utilization"] * rail_util +
            REWARD_WEIGHTS["safety_compliance"] * safety
        )

        return {
            "berth_utilization": round(berth_util, 4),
            "crane_productivity": round(crane_prod, 4),
            "vessel_turnaround": round(vessel_turn, 4),
            "yard_efficiency": round(yard_eff, 4),
            "truck_turnaround": round(truck_turn, 4),
            "rail_utilization": round(rail_util, 4),
            "safety_compliance": round(safety, 4),
            "weighted_total": round(weighted, 4),
        }

    def compute_final_reward(self) -> Dict[str, Any]:
        if not self.step_rewards:
            # Compute at least one step reward
            self.step_rewards.append(self.compute_step_reward())

        mean_reward = sum(s["weighted_total"] for s in self.step_rewards) / len(self.step_rewards)
        mean_reward = max(0.0, min(1.0, mean_reward))

        # Component averages
        components = {}
        for key in REWARD_WEIGHTS:
            vals = [s.get(key, 0.0) for s in self.step_rewards]
            components[f"avg_{key}"] = round(sum(vals) / len(vals), 4)

        vessels_completed = sum(1 for v in self.vessels.values()
                               if v.status == VesselStatus.DEPARTED)
        total_moves = sum(c.total_moves for c in self.cranes.values())

        return {
            "total_reward": round(mean_reward, 4),
            "num_steps": len(self.step_rewards),
            "vessels_completed": vessels_completed,
            "vessels_total": len(self.vessels),
            "total_crane_moves": total_moves,
            "trains_departed": len(self.trains_departed),
            "safety_violations": self.safety_violations,
            "clock": round(self.clock, 1),
            **components,
        }

    def _count_safety_violations(self) -> int:
        violations = 0
        for yb in self.yard_blocks.values():
            # Reefer containers in non-power blocks
            if not yb.has_power_points and yb.reefer_occupied > 0:
                violations += 1
            # Hazmat containers in non-hazmat blocks
            hazmat_in_block = yb.containers_by_type.get("hazmat", 0)
            if not yb.hazmat_zone and hazmat_in_block > 0:
                violations += 1
            # Over-capacity
            if yb.current_occupancy > yb.total_capacity:
                violations += 1
        self.safety_violations = violations
        return violations
