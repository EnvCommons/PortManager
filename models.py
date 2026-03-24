"""
Data models for the Container Port Manager simulation.

Defines enums, dataclasses, and constants for a medium-large container terminal
with 4 berths, 12 STS cranes, 20 yard blocks, 8 gate lanes, and 4 rail tracks.

Parameter references (verified against industry standards and academic sources):
  Crane throughput: 25-40 moves/hour per STS crane, baseline 30 GMPH
    [IAPH, "Study on Productivity and Key Indicators of Container Terminals", 2019
     (~26 GMPH global average); Liftech Consultants, "Quay Crane Productivity", 2002
     (advanced systems 45-55 GMPH)]
  Vessel turnaround: Feeder 12-24h, Panamax 24-48h, Post-Panamax 36-72h, ULCV 48-96h
    [FreightAmigo, "Comparing Unloading Times of Different Container Ships", 2025
     (adapted planning ranges; source values differ slightly by vessel class)]
  Vessel TEU capacities: Feeder 1K-3K, Panamax 3K-5.1K, Post-Panamax 5K-10K, ULCV 14K-20K
    [Rodrigue, "The Geography of Transport Systems", 6th Edition, Routledge, 2024]
  Vessel drafts: Feeder 8-11m, Panamax 11-13.5m, Post-Panamax 13-15.5m, ULCV 15-17.5m
    [Rodrigue, "The Geography of Transport Systems", 2024; Wikipedia Panamax
     (canal limit 12.04m); operational draft ranges exceed canal constraints]
  Wind halt threshold: 20 m/s (~40 knots) for STS crane operations
    [ISO 4302:2016 (20 m/s general crane limit); Scarlet-Tech,
     "Understanding Crane Wind Speed Limits"; Wikipedia Container Crane (~22 m/s)]
  Crane MTBF: 500+ operational hours for significant failures
    [Jo and Kim, "Key Performance Indicator Development for Ship-to-Shore Crane
     Performance Assessment", JMSE 2020, 8(1):6 (MMBF ~1,805 moves; 500 operational
     hours is an estimated conversion at ~3.6 moves/min utilization)]
  Crane repair time: 2-8 hours for minor to moderate failures
    [Global Rigging, "Five Common Ship-to-Shore Crane Problems", 2023
     (general overview); major structural repairs can take days to weeks]
  Yard block dimensions: 6 rows x 30 bays x 5 tiers, 70% effective capacity (RTG config)
    [Port Economics PEMP (Notteboom, Pallis, Rodrigue), "Configuration of Container
     Yards" (general layout); 6x30x5 is within standard RTG industry ranges;
     65-70% effective capacity is a common operational planning assumption]
  Gate throughput: ~30 trucks/hour/lane (semi-automated with OCR)
    [Moszyk, Deja, and Dobrzynski, "Automation of the Road Gate Operations Process
     at the Container Terminal", Sustainability 2021, 13(11):6291 (DCT Gdansk);
     ~30 trucks/hr/lane corresponds to ~2 min processing per truck]
  Rail train capacity: 30 cars x 4 TEU/car = 120 TEU (European/shuttle train standard)
    [Containerlift, "Introduction to Intermodal Freight Trains", 2025; 30 wagons
     typical for European shuttle trains; 4 TEU/wagon assumes larger wagon types;
     total 120 TEU is within realistic range]
  Tide windows: Semi-diurnal M2 period 12.42 hours, ~4h usable high-tide window
    [NOAA, "Types and Causes of Tidal Cycles"]
  Vessel schedule reliability: ~50-55% on-time
    [Sea-Intelligence, Global Liner Performance Report, 2024]
  Customs inspection rate: 3-5% of containers (US CBP standard), 15% for elevated alerts
    [CBO, "Scanning and Imaging Shipping Containers Overseas", 2016 (3-5% routine);
     15% elevated inspection rate is a simulation parameter]
  Reefer percentage: ~5-6% of global loaded TEU
    [Drewry Maritime Research (~5% of global loaded TEU); Allied Market Research,
     "Reefer Container Market" (market sizing)]
  Hazmat segregation: IMDG Code minimum 3m horizontal separation
    [IMO, International Maritime Dangerous Goods Code, Segregation Level 1 "Away from"
     (3m minimum horizontal for breakbulk; simplified for yard operations)]
  Container exchange fraction: 40-80% of TEU capacity per port call
    [Industry operational planning parameter; range varies by port role — see UNCTAD,
     Review of Maritime Transport, 2022 for port performance context]
  Berth dimensions: 300-450m length, 12-18m draft for medium-large terminal
    [Port Economics PEMP, "Terminal Depth at Selected Ports"]
"""

import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class VesselType(str, Enum):
    FEEDER = "feeder"
    PANAMAX = "panamax"
    POST_PANAMAX = "post_panamax"
    ULCV = "ulcv"


class VesselStatus(str, Enum):
    SCHEDULED = "scheduled"
    WAITING = "waiting"
    BERTHED = "berthed"
    DEPARTING = "departing"
    DEPARTED = "departed"


class CraneStatus(str, Enum):
    IDLE = "idle"
    WORKING = "working"
    MOVING = "moving"
    BREAKDOWN = "breakdown"


class ContainerType(str, Enum):
    DRY = "dry"
    REEFER = "reefer"
    HAZMAT = "hazmat"


class DisruptionType(str, Enum):
    STORM = "storm"
    LABOR_STRIKE = "labor_strike"
    CUSTOMS_HOLD = "customs_hold"
    EQUIPMENT_BREAKDOWN = "equipment_breakdown"
    VESSEL_DELAY = "vessel_delay"
    HIGH_PRIORITY_CARGO = "high_priority_cargo"


class GateDirection(str, Enum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Berth:
    berth_id: str
    length_m: float
    draft_m: float
    max_cranes: int
    vessel_id: Optional[str] = None
    occupied_since: Optional[float] = None


@dataclass
class Crane:
    crane_id: str
    berth_id: str
    status: CraneStatus = CraneStatus.IDLE
    vessel_id: Optional[str] = None
    moves_per_hour: float = 30.0
    actual_moves_this_hour: int = 0
    total_moves: int = 0
    operational_hours: float = 0.0
    next_breakdown_hour: Optional[float] = None
    repair_end_hour: Optional[float] = None
    move_end_hour: Optional[float] = None
    target_berth_id: Optional[str] = None


@dataclass
class Vessel:
    vessel_id: str
    vessel_type: VesselType
    teu_capacity: int
    import_containers: int
    export_containers: int
    reefer_count: int
    hazmat_count: int
    draft_required_m: float
    min_cranes: int
    max_cranes: int
    target_turnaround_hours: float
    scheduled_arrival: float
    priority: int = 0  # higher = more important
    actual_arrival: Optional[float] = None
    berth_id: Optional[str] = None
    status: VesselStatus = VesselStatus.SCHEDULED
    containers_unloaded: int = 0
    containers_loaded: int = 0
    berthing_time: Optional[float] = None
    departure_time: Optional[float] = None
    cranes_assigned: List[str] = field(default_factory=list)
    yard_blocks_import: List[str] = field(default_factory=list)
    yard_blocks_export: List[str] = field(default_factory=list)

    @property
    def total_moves(self) -> int:
        return self.import_containers + self.export_containers

    @property
    def remaining_moves(self) -> int:
        return (self.import_containers - self.containers_unloaded) + \
               (self.export_containers - self.containers_loaded)

    @property
    def is_deep_draft(self) -> bool:
        return self.draft_required_m > 14.0


@dataclass
class YardBlock:
    block_id: str
    rows: int = 6
    bays: int = 30
    tiers: int = 5
    total_capacity: int = 900
    effective_capacity: int = 630
    current_occupancy: int = 0
    reefer_slots: int = 45
    reefer_occupied: int = 0
    has_power_points: bool = False
    hazmat_zone: bool = False
    containers_by_vessel: Dict[str, int] = field(default_factory=dict)
    containers_by_type: Dict[str, int] = field(default_factory=dict)
    customs_held: int = 0


@dataclass
class GateLane:
    lane_id: str
    direction: GateDirection
    throughput_per_hour: int = 30
    current_queue: int = 0
    trucks_processed_this_hour: int = 0
    total_trucks_processed: int = 0


@dataclass
class RailTrack:
    track_id: str
    max_cars: int = 30
    teu_per_car: int = 4
    max_teu: int = 120
    current_load_teu: int = 0
    scheduled_departure: Optional[float] = None
    loading_from_blocks: List[str] = field(default_factory=list)
    departed: bool = False
    departure_actual: Optional[float] = None


@dataclass
class Disruption:
    disruption_id: str
    disruption_type: DisruptionType
    start_hour: float
    end_hour: float
    severity: float  # 0.0-1.0
    details: Dict[str, Any] = field(default_factory=dict)
    active: bool = False
    resolved: bool = False
    agent_action: Optional[str] = None


@dataclass(order=True)
class PortEvent:
    time: float
    sequence: int = field(compare=True)
    event_type: str = field(compare=False)
    entity_id: Optional[str] = field(default=None, compare=False)
    details: Optional[Dict[str, Any]] = field(default=None, compare=False)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PLANNING_HORIZON = 168.0  # 1 week in hours

VESSEL_CONFIGS = {
    VesselType.FEEDER: {
        "teu_range": (1000, 3000),
        "min_cranes": 1,
        "max_cranes": 2,
        "target_turnaround": (12, 24),
        "draft_range": (8.0, 11.0),
        "exchange_fraction": (0.40, 0.70),
    },
    VesselType.PANAMAX: {
        "teu_range": (3000, 5100),
        "min_cranes": 2,
        "max_cranes": 3,
        "target_turnaround": (24, 48),
        "draft_range": (11.0, 13.5),
        "exchange_fraction": (0.50, 0.75),
    },
    VesselType.POST_PANAMAX: {
        "teu_range": (5000, 10000),
        "min_cranes": 3,
        "max_cranes": 5,
        "target_turnaround": (36, 72),
        "draft_range": (13.0, 15.5),
        "exchange_fraction": (0.50, 0.80),
    },
    VesselType.ULCV: {
        "teu_range": (14000, 20000),
        "min_cranes": 5,
        "max_cranes": 8,
        "target_turnaround": (48, 96),
        "draft_range": (15.0, 17.5),
        "exchange_fraction": (0.60, 0.80),
    },
}

BERTH_CONFIGS = [
    {"berth_id": "B1", "length_m": 350.0, "draft_m": 14.0, "max_cranes": 3},
    {"berth_id": "B2", "length_m": 400.0, "draft_m": 16.0, "max_cranes": 4},
    {"berth_id": "B3", "length_m": 450.0, "draft_m": 18.0, "max_cranes": 4},
    {"berth_id": "B4", "length_m": 300.0, "draft_m": 12.0, "max_cranes": 3},
]

# Initial crane distribution: 3 per berth = 12 total
CRANE_DISTRIBUTION = {
    "B1": ["QC01", "QC02", "QC03"],
    "B2": ["QC04", "QC05", "QC06"],
    "B3": ["QC07", "QC08", "QC09"],
    "B4": ["QC10", "QC11", "QC12"],
}

NUM_YARD_BLOCKS = 20
HAZMAT_BLOCKS = {"YB19", "YB20"}  # Last 2 blocks are hazmat zones
REEFER_POWER_BLOCKS = {"YB01", "YB02", "YB03", "YB04"}  # First 4 have power

NUM_GATE_LANES_IN = 4
NUM_GATE_LANES_OUT = 4
NUM_RAIL_TRACKS = 4

# Disruption thresholds
WIND_CRANE_HALT_KNOTS = 40.0
WIND_GATE_REDUCTION_KNOTS = 30.0
STORM_GATE_THROUGHPUT_FACTOR = 0.50

# Crane parameters
CRANE_MTBF_HOURS = 500.0
CRANE_REPAIR_RANGE = (2.0, 8.0)
CRANE_MOVE_DURATION = 1.0  # hours to relocate between berths
CRANE_BASELINE_MPH = 30.0  # moves per hour benchmark
CRANE_EFFICIENCY_NOISE = (0.85, 1.15)  # multiplicative noise range

# Default vessel mix (fraction per type)
DEFAULT_VESSEL_MIX = {
    VesselType.FEEDER: 0.30,
    VesselType.PANAMAX: 0.35,
    VesselType.POST_PANAMAX: 0.25,
    VesselType.ULCV: 0.10,
}

# Reward weights
REWARD_WEIGHTS = {
    "berth_utilization": 0.20,
    "crane_productivity": 0.20,
    "vessel_turnaround": 0.20,
    "yard_efficiency": 0.15,
    "truck_turnaround": 0.10,
    "rail_utilization": 0.10,
    "safety_compliance": 0.05,
}
