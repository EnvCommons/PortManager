"""
Task generation for the Container Port Manager environment.

Produces 40 tasks (30 train, 10 test) by varying seed, scenario type,
vessel count, vessel mix, disruption profile, cargo composition, and
delay factors.
"""

from typing import Any, Dict, List

from models import VesselType, DisruptionType


def _make_task(
    task_id: str,
    seed: int,
    scenario_type: str,
    num_vessels: int,
    vessel_mix: Dict[str, float],
    disruptions: List[Dict[str, Any]],
    yard_initial_occupancy: float,
    reefer_fraction: float,
    hazmat_fraction: float,
    vessel_delay_factor: float,
    description: str,
) -> Dict[str, Any]:
    return {
        "id": task_id,
        "seed": seed,
        "scenario_type": scenario_type,
        "num_vessels": num_vessels,
        "vessel_mix": vessel_mix,
        "disruptions": disruptions,
        "yard_initial_occupancy": yard_initial_occupancy,
        "reefer_fraction": reefer_fraction,
        "hazmat_fraction": hazmat_fraction,
        "vessel_delay_factor": vessel_delay_factor,
        "description": description,
    }


# Default vessel mixes
BALANCED_MIX = {
    "feeder": 0.30, "panamax": 0.35, "post_panamax": 0.25, "ulcv": 0.10,
}
MEGA_MIX = {
    "feeder": 0.15, "panamax": 0.20, "post_panamax": 0.30, "ulcv": 0.35,
}
FEEDER_HEAVY_MIX = {
    "feeder": 0.50, "panamax": 0.30, "post_panamax": 0.15, "ulcv": 0.05,
}

# Common disruption profiles
CALM_DISRUPTIONS = [
    {"type": "equipment_breakdown", "start_hour": 72.0, "end_hour": 76.0, "severity": 0.3,
     "details": {"crane_id": "QC05"}},
]

STORM_DISRUPTIONS = [
    {"type": "storm", "start_hour": 36.0, "end_hour": 48.0, "severity": 0.9,
     "details": {"wind_knots": 50.0}},
    {"type": "storm", "start_hour": 120.0, "end_hour": 128.0, "severity": 0.7,
     "details": {"wind_knots": 45.0}},
]

STRIKE_DISRUPTIONS = [
    {"type": "labor_strike", "start_hour": 72.0, "end_hour": 120.0, "severity": 0.5,
     "details": {"productivity_factor": 0.5}},
]

FULL_STRIKE_DISRUPTIONS = [
    {"type": "labor_strike", "start_hour": 60.0, "end_hour": 96.0, "severity": 1.0,
     "details": {"productivity_factor": 0.0}},
]

BREAKDOWN_DISRUPTIONS = [
    {"type": "equipment_breakdown", "start_hour": 24.0, "end_hour": 30.0, "severity": 0.5,
     "details": {"crane_id": "QC02"}},
    {"type": "equipment_breakdown", "start_hour": 60.0, "end_hour": 68.0, "severity": 0.5,
     "details": {"crane_id": "QC07"}},
    {"type": "equipment_breakdown", "start_hour": 100.0, "end_hour": 105.0, "severity": 0.5,
     "details": {"crane_id": "QC11"}},
    {"type": "equipment_breakdown", "start_hour": 130.0, "end_hour": 136.0, "severity": 0.5,
     "details": {"crane_id": "QC04"}},
]

CUSTOMS_DISRUPTIONS = [
    {"type": "customs_hold", "start_hour": 0.0, "end_hour": 168.0, "severity": 0.8,
     "details": {"inspection_rate": 0.15}},
]

PERFECT_STORM_DISRUPTIONS = [
    {"type": "storm", "start_hour": 48.0, "end_hour": 60.0, "severity": 0.9,
     "details": {"wind_knots": 50.0}},
    {"type": "labor_strike", "start_hour": 80.0, "end_hour": 128.0, "severity": 0.5,
     "details": {"productivity_factor": 0.5}},
    {"type": "equipment_breakdown", "start_hour": 36.0, "end_hour": 42.0, "severity": 0.5,
     "details": {"crane_id": "QC08"}},
]


def generate_tasks() -> Dict[str, List[Dict[str, Any]]]:
    train_tasks: List[Dict[str, Any]] = []
    test_tasks: List[Dict[str, Any]] = []

    # ----------------------------------------------------------------
    # Training tasks (30): 6 scenario types x 5 seeds each
    # ----------------------------------------------------------------

    # Calm week (5): normal operations, 1 minor breakdown
    for i, seed in enumerate([101, 102, 103, 104, 105]):
        nv = [9, 8, 10, 9, 8][i]
        train_tasks.append(_make_task(
            task_id=f"port_train_{len(train_tasks):03d}",
            seed=seed, scenario_type="calm_week", num_vessels=nv,
            vessel_mix=BALANCED_MIX, disruptions=CALM_DISRUPTIONS,
            yard_initial_occupancy=0.40, reefer_fraction=0.05,
            hazmat_fraction=0.02, vessel_delay_factor=0.2,
            description="Normal operations with minor disruptions",
        ))

    # Storm season (5): 2 major storms
    for i, seed in enumerate([201, 202, 203, 204, 205]):
        nv = [9, 10, 8, 9, 10][i]
        train_tasks.append(_make_task(
            task_id=f"port_train_{len(train_tasks):03d}",
            seed=seed, scenario_type="storm_season", num_vessels=nv,
            vessel_mix=BALANCED_MIX, disruptions=STORM_DISRUPTIONS,
            yard_initial_occupancy=0.45, reefer_fraction=0.05,
            hazmat_fraction=0.02, vessel_delay_factor=0.3,
            description="Two major storms causing crane shutdowns",
        ))

    # Labor dispute (5): partial strike mid-week
    for i, seed in enumerate([301, 302, 303, 304, 305]):
        nv = [9, 8, 10, 9, 8][i]
        train_tasks.append(_make_task(
            task_id=f"port_train_{len(train_tasks):03d}",
            seed=seed, scenario_type="labor_dispute", num_vessels=nv,
            vessel_mix=BALANCED_MIX, disruptions=STRIKE_DISRUPTIONS,
            yard_initial_occupancy=0.45, reefer_fraction=0.05,
            hazmat_fraction=0.02, vessel_delay_factor=0.3,
            description="Partial labor strike reduces productivity 50%",
        ))

    # Peak season (5): 30% more vessel arrivals
    for i, seed in enumerate([401, 402, 403, 404, 405]):
        nv = [13, 12, 14, 13, 12][i]
        train_tasks.append(_make_task(
            task_id=f"port_train_{len(train_tasks):03d}",
            seed=seed, scenario_type="peak_season", num_vessels=nv,
            vessel_mix=BALANCED_MIX, disruptions=CALM_DISRUPTIONS,
            yard_initial_occupancy=0.55, reefer_fraction=0.05,
            hazmat_fraction=0.02, vessel_delay_factor=0.4,
            description="Peak season with 30% more vessel arrivals",
        ))

    # Equipment aging (5): higher breakdown rates
    for i, seed in enumerate([501, 502, 503, 504, 505]):
        nv = [9, 8, 10, 9, 8][i]
        train_tasks.append(_make_task(
            task_id=f"port_train_{len(train_tasks):03d}",
            seed=seed, scenario_type="equipment_aging", num_vessels=nv,
            vessel_mix=BALANCED_MIX, disruptions=BREAKDOWN_DISRUPTIONS,
            yard_initial_occupancy=0.45, reefer_fraction=0.05,
            hazmat_fraction=0.02, vessel_delay_factor=0.3,
            description="Aging equipment with frequent crane breakdowns",
        ))

    # Mixed cargo (5): high reefer + hazmat
    for i, seed in enumerate([601, 602, 603, 604, 605]):
        nv = [9, 8, 10, 9, 8][i]
        train_tasks.append(_make_task(
            task_id=f"port_train_{len(train_tasks):03d}",
            seed=seed, scenario_type="mixed_cargo", num_vessels=nv,
            vessel_mix=BALANCED_MIX, disruptions=CALM_DISRUPTIONS,
            yard_initial_occupancy=0.45, reefer_fraction=0.15,
            hazmat_fraction=0.08, vessel_delay_factor=0.3,
            description="High proportion of reefer and hazmat cargo",
        ))

    # ----------------------------------------------------------------
    # Test tasks (10): 4 scenario types
    # ----------------------------------------------------------------

    # Perfect storm (3): storm + labor + delays
    for i, seed in enumerate([701, 702, 703]):
        nv = [11, 10, 12][i]
        test_tasks.append(_make_task(
            task_id=f"port_test_{len(test_tasks):03d}",
            seed=seed, scenario_type="perfect_storm", num_vessels=nv,
            vessel_mix=BALANCED_MIX, disruptions=PERFECT_STORM_DISRUPTIONS,
            yard_initial_occupancy=0.50, reefer_fraction=0.08,
            hazmat_fraction=0.04, vessel_delay_factor=0.6,
            description="Storm + labor dispute + vessel delays + equipment failure",
        ))

    # Mega vessel week (3): multiple ULCVs
    for i, seed in enumerate([801, 802, 803]):
        nv = [10, 11, 10][i]
        test_tasks.append(_make_task(
            task_id=f"port_test_{len(test_tasks):03d}",
            seed=seed, scenario_type="mega_vessel_week", num_vessels=nv,
            vessel_mix=MEGA_MIX, disruptions=CALM_DISRUPTIONS,
            yard_initial_occupancy=0.50, reefer_fraction=0.06,
            hazmat_fraction=0.03, vessel_delay_factor=0.4,
            description="Multiple ULCV arrivals competing for deep-draft berths",
        ))

    # Customs crackdown (2): 15% inspection rate
    for i, seed in enumerate([901, 902]):
        nv = [9, 10][i]
        test_tasks.append(_make_task(
            task_id=f"port_test_{len(test_tasks):03d}",
            seed=seed, scenario_type="customs_crackdown", num_vessels=nv,
            vessel_mix=BALANCED_MIX, disruptions=CUSTOMS_DISRUPTIONS,
            yard_initial_occupancy=0.55, reefer_fraction=0.05,
            hazmat_fraction=0.02, vessel_delay_factor=0.3,
            description="Elevated customs inspections blocking yard space",
        ))

    # Cascade crisis (2): multiple delayed vessels arriving simultaneously
    for i, seed in enumerate([1001, 1002]):
        nv = [13, 14][i]
        test_tasks.append(_make_task(
            task_id=f"port_test_{len(test_tasks):03d}",
            seed=seed, scenario_type="cascade_crisis", num_vessels=nv,
            vessel_mix=BALANCED_MIX,
            disruptions=[
                {"type": "storm", "start_hour": 20.0, "end_hour": 28.0, "severity": 0.8,
                 "details": {"wind_knots": 45.0}},
                {"type": "equipment_breakdown", "start_hour": 90.0, "end_hour": 98.0,
                 "severity": 0.5, "details": {"crane_id": "QC03"}},
            ],
            yard_initial_occupancy=0.60, reefer_fraction=0.06,
            hazmat_fraction=0.03, vessel_delay_factor=0.7,
            description="Multiple delayed vessels causing yard overflow and cascading congestion",
        ))

    return {"train": train_tasks, "test": test_tasks}


ALL_TASKS = generate_tasks()
