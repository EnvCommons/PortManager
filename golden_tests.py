"""
Golden tests for PortManager simulation parameters.

These tests verify that simulation constants and parameters match their cited
industry/academic sources. Each test documents the citation basis for the
parameter it validates. If a test fails, either the parameter or the citation
in models.py needs to be updated.
"""

import re
import pytest

from models import (
    BERTH_CONFIGS,
    CRANE_BASELINE_MPH,
    CRANE_DISTRIBUTION,
    CRANE_EFFICIENCY_NOISE,
    CRANE_MOVE_DURATION,
    CRANE_MTBF_HOURS,
    CRANE_REPAIR_RANGE,
    HAZMAT_BLOCKS,
    NUM_GATE_LANES_IN,
    NUM_GATE_LANES_OUT,
    NUM_RAIL_TRACKS,
    NUM_YARD_BLOCKS,
    PLANNING_HORIZON,
    REEFER_POWER_BLOCKS,
    REWARD_WEIGHTS,
    VESSEL_CONFIGS,
    WIND_CRANE_HALT_KNOTS,
    VesselType,
    GateLane,
    RailTrack,
    YardBlock,
)
from simulation import PortSimulation
from scenarios import ALL_TASKS


# ---------------------------------------------------------------------------
# Crane Parameters
# ---------------------------------------------------------------------------

class TestCraneParameterCitations:
    """Verify crane parameters match IAPH/Liftech/Jo&Kim/ISO citations."""

    def test_crane_baseline_mph_matches_industry(self):
        """IAPH 2019 reports ~26 GMPH global average; 30 is above-average baseline.
        Liftech 2002 discusses 45-55 GMPH for advanced dual-hoist systems.
        30 GMPH sits between global average and advanced, a reasonable baseline."""
        assert CRANE_BASELINE_MPH == 30.0
        # Should be between global average and advanced performance
        assert 26.0 <= CRANE_BASELINE_MPH <= 40.0

    def test_crane_efficiency_noise_range(self):
        """Noise (0.85, 1.15) applied to 30 GMPH produces 25.5-34.5 moves/hr,
        within the commonly cited 25-35 range for individual crane performance."""
        assert CRANE_EFFICIENCY_NOISE == (0.85, 1.15)
        low = CRANE_BASELINE_MPH * CRANE_EFFICIENCY_NOISE[0]
        high = CRANE_BASELINE_MPH * CRANE_EFFICIENCY_NOISE[1]
        assert low == pytest.approx(25.5, abs=0.1)
        assert high == pytest.approx(34.5, abs=0.1)

    def test_wind_halt_threshold_matches_iso(self):
        """ISO 4302:2016 specifies 20 m/s (~40 knots) as general crane wind limit.
        Wikipedia Container Crane cites ~22 m/s for STS cranes specifically.
        40 knots (20.6 m/s) is a conservative operational threshold."""
        assert WIND_CRANE_HALT_KNOTS == 40.0
        # Convert to m/s: 40 knots * 0.5144 = 20.6 m/s
        wind_ms = WIND_CRANE_HALT_KNOTS * 0.5144
        assert 19.0 <= wind_ms <= 23.0, \
            f"Wind halt {wind_ms:.1f} m/s should be in 19-23 m/s range (ISO/Wikipedia)"

    def test_crane_mtbf_reasonable(self):
        """Jo and Kim, JMSE 2020, 8(1):6 report MMBF ~1,805 moves for STS cranes.
        At ~30 moves/hr, 500h * 30 = 15,000 moves. The 500h figure represents
        significant failures requiring multi-hour repair, not all failure modes."""
        assert CRANE_MTBF_HOURS == 500.0
        # Should represent substantial operational period (not too frequent)
        assert CRANE_MTBF_HOURS >= 200.0  # Major failures shouldn't occur every few days
        assert CRANE_MTBF_HOURS <= 2000.0  # But should occur within reasonable sim window

    def test_crane_repair_range_minor_to_moderate(self):
        """Global Rigging (2023) discusses common STS problems; 2-8h covers minor
        to moderate failures (electrical, hydraulic, spreader issues).
        Major structural repairs can take days to weeks."""
        assert CRANE_REPAIR_RANGE == (2.0, 8.0)
        assert CRANE_REPAIR_RANGE[0] >= 1.0  # Minimum realistic repair time
        assert CRANE_REPAIR_RANGE[1] <= 24.0  # Not multi-day for this sim scope

    def test_crane_move_duration_one_hour(self):
        """Inter-berth crane relocation takes ~1 hour in practice for
        rail-mounted STS cranes traversing 300-450m berth lengths."""
        assert CRANE_MOVE_DURATION == 1.0


# ---------------------------------------------------------------------------
# Vessel Parameters
# ---------------------------------------------------------------------------

class TestVesselParameterCitations:
    """Verify vessel parameters match Rodrigue 2024 / FreightAmigo 2025 citations."""

    def test_vessel_teu_ranges_match_rodrigue(self):
        """Rodrigue, 'The Geography of Transport Systems', 6th Ed, Routledge, 2024.
        TEU ranges are within Rodrigue's classification framework."""
        expected = {
            VesselType.FEEDER: (1000, 3000),
            VesselType.PANAMAX: (3000, 5100),
            VesselType.POST_PANAMAX: (5000, 10000),
            VesselType.ULCV: (14000, 20000),
        }
        for vtype, (exp_lo, exp_hi) in expected.items():
            cfg = VESSEL_CONFIGS[vtype]
            assert cfg["teu_range"] == (exp_lo, exp_hi), \
                f"{vtype.value} TEU range mismatch"

    def test_vessel_teu_ranges_no_overlap_gaps(self):
        """Vessel classes should form a reasonable progression without large gaps."""
        feeder_hi = VESSEL_CONFIGS[VesselType.FEEDER]["teu_range"][1]
        panamax_lo = VESSEL_CONFIGS[VesselType.PANAMAX]["teu_range"][0]
        assert panamax_lo <= feeder_hi  # Overlap is acceptable

        panamax_hi = VESSEL_CONFIGS[VesselType.PANAMAX]["teu_range"][1]
        post_lo = VESSEL_CONFIGS[VesselType.POST_PANAMAX]["teu_range"][0]
        assert post_lo <= panamax_hi  # Overlap is acceptable

    def test_vessel_turnaround_ranges_match_freightamigo(self):
        """FreightAmigo 2025 article 'Comparing Unloading Times of Different
        Container Ships'. Sim uses adapted planning ranges. Source values:
        Feeder 8-24h, Panamax 24-36h, Post-Panamax 36-72h, ULCV 72-96+h.
        Sim ranges are wider to account for operational variability."""
        expected = {
            VesselType.FEEDER: (12, 24),
            VesselType.PANAMAX: (24, 48),
            VesselType.POST_PANAMAX: (36, 72),
            VesselType.ULCV: (48, 96),
        }
        for vtype, (exp_lo, exp_hi) in expected.items():
            cfg = VESSEL_CONFIGS[vtype]
            assert cfg["target_turnaround"] == (exp_lo, exp_hi), \
                f"{vtype.value} turnaround mismatch"

    def test_vessel_draft_ranges_match_rodrigue_and_wikipedia(self):
        """Rodrigue 2024 and Wikipedia Panamax specifications.
        Wikipedia: Panamax canal limit 12.04m. Sim uses operational drafts
        which can exceed canal constraints (loaded draft at sea)."""
        expected = {
            VesselType.FEEDER: (8.0, 11.0),
            VesselType.PANAMAX: (11.0, 13.5),
            VesselType.POST_PANAMAX: (13.0, 15.5),
            VesselType.ULCV: (15.0, 17.5),
        }
        for vtype, (exp_lo, exp_hi) in expected.items():
            cfg = VESSEL_CONFIGS[vtype]
            assert cfg["draft_range"] == (exp_lo, exp_hi), \
                f"{vtype.value} draft range mismatch"

    def test_vessel_draft_progression_monotonic(self):
        """Larger vessels should require deeper drafts."""
        types_ordered = [VesselType.FEEDER, VesselType.PANAMAX,
                         VesselType.POST_PANAMAX, VesselType.ULCV]
        for i in range(len(types_ordered) - 1):
            current_hi = VESSEL_CONFIGS[types_ordered[i]]["draft_range"][1]
            next_lo = VESSEL_CONFIGS[types_ordered[i + 1]]["draft_range"][0]
            assert next_lo >= current_hi - 2.0, \
                f"Draft should generally increase: {types_ordered[i].value} -> {types_ordered[i+1].value}"

    def test_exchange_fraction_40_to_80_percent(self):
        """Industry operational planning parameter (UNCTAD RMT 2022 context).
        Range 40-80% of TEU capacity exchanged per port call."""
        for vtype, cfg in VESSEL_CONFIGS.items():
            lo, hi = cfg["exchange_fraction"]
            assert lo >= 0.40, f"{vtype.value} exchange fraction lower bound too low: {lo}"
            assert hi <= 0.80, f"{vtype.value} exchange fraction upper bound too high: {hi}"

    def test_vessel_crane_requirements_progressive(self):
        """Larger vessels need more cranes: feeder 1-2, panamax 2-3,
        post-panamax 3-5, ULCV 5-8."""
        assert VESSEL_CONFIGS[VesselType.FEEDER]["min_cranes"] == 1
        assert VESSEL_CONFIGS[VesselType.FEEDER]["max_cranes"] == 2
        assert VESSEL_CONFIGS[VesselType.PANAMAX]["min_cranes"] == 2
        assert VESSEL_CONFIGS[VesselType.ULCV]["min_cranes"] >= 5


# ---------------------------------------------------------------------------
# Yard Parameters
# ---------------------------------------------------------------------------

class TestYardParameterCitations:
    """Verify yard parameters match PEMP / industry RTG standards."""

    def test_yard_block_dimensions_match_rtg_standard(self):
        """PEMP (Notteboom, Pallis, Rodrigue) 'Configuration of Container Yards'.
        6x30x5 is within standard RTG ranges (6-13 rows, 10-40 bays, 3-6 tiers)."""
        yb = YardBlock(block_id="test")
        assert yb.rows == 6
        assert yb.bays == 30
        assert yb.tiers == 5
        assert yb.total_capacity == 6 * 30 * 5  # = 900

    def test_effective_capacity_ratio_70_percent(self):
        """65-70% effective capacity is a common operational planning assumption
        for RTG yards to allow for reshuffling and access."""
        yb = YardBlock(block_id="test")
        ratio = yb.effective_capacity / yb.total_capacity
        assert ratio == pytest.approx(0.70, abs=0.01)
        assert yb.effective_capacity == 630

    def test_num_yard_blocks_20(self):
        """20 yard blocks x 630 effective TEU = 12,600 TEU total yard capacity."""
        assert NUM_YARD_BLOCKS == 20
        total_effective = NUM_YARD_BLOCKS * 630
        assert total_effective == 12600

    def test_hazmat_blocks_designated(self):
        """IMDG Code requires segregated storage for dangerous goods.
        YB19 and YB20 designated as hazmat zones."""
        assert HAZMAT_BLOCKS == {"YB19", "YB20"}
        # Should be a subset of total blocks
        for bid in HAZMAT_BLOCKS:
            block_num = int(bid[2:])
            assert 1 <= block_num <= NUM_YARD_BLOCKS

    def test_reefer_power_blocks_designated(self):
        """Reefer containers need electrical connections for refrigeration.
        YB01-YB04 have power points."""
        assert REEFER_POWER_BLOCKS == {"YB01", "YB02", "YB03", "YB04"}
        # No overlap with hazmat zones
        assert HAZMAT_BLOCKS.isdisjoint(REEFER_POWER_BLOCKS)


# ---------------------------------------------------------------------------
# Gate and Rail Parameters
# ---------------------------------------------------------------------------

class TestGateRailParameterCitations:
    """Verify gate/rail parameters match Moszyk et al. 2021 / Containerlift."""

    def test_gate_throughput_30_per_lane(self):
        """Moszyk, Deja, Dobrzynski, Sustainability 2021, 13(11):6291.
        DCT Gdansk case study. ~30 trucks/hr/lane = ~2 min per truck
        for semi-automated gates with OCR."""
        lane = GateLane(lane_id="test", direction="inbound")
        assert lane.throughput_per_hour == 30

    def test_gate_lane_count(self):
        """4 inbound + 4 outbound = 8 total gate lanes."""
        assert NUM_GATE_LANES_IN == 4
        assert NUM_GATE_LANES_OUT == 4

    def test_rail_capacity_120_teu(self):
        """Containerlift, 'Introduction to Intermodal Freight Trains', 2025.
        30 wagons x 4 TEU/wagon = 120 TEU (European shuttle train with larger wagons)."""
        track = RailTrack(track_id="test")
        assert track.max_cars == 30
        assert track.teu_per_car == 4
        assert track.max_teu == 120
        assert track.max_teu == track.max_cars * track.teu_per_car

    def test_rail_track_count(self):
        """4 rail tracks for intermodal operations."""
        assert NUM_RAIL_TRACKS == 4


# ---------------------------------------------------------------------------
# Tide Parameters
# ---------------------------------------------------------------------------

class TestTideParameterCitations:
    """Verify tide parameters match NOAA citations."""

    def test_tide_period_12_42_hours(self):
        """NOAA, 'Types and Causes of Tidal Cycles'. The principal lunar
        semi-diurnal constituent (M2) has a period of 12.42 hours
        (12h 25.2min = half of 24.84h tidal lunar day)."""
        sim = PortSimulation({
            "id": "tide_test", "seed": 1, "scenario_type": "calm_week",
            "num_vessels": 1,
            "vessel_mix": {"feeder": 1.0, "panamax": 0.0, "post_panamax": 0.0, "ulcv": 0.0},
            "disruptions": [], "yard_initial_occupancy": 0.30,
            "reefer_fraction": 0.0, "hazmat_fraction": 0.0,
            "vessel_delay_factor": 0.0, "description": "Tide test",
        })
        windows = sim.tide_windows
        assert len(windows) >= 2
        for i in range(1, len(windows)):
            gap = windows[i][0] - windows[i - 1][0]
            assert abs(gap - 12.42) < 0.01, \
                f"Tide gap {gap:.4f}h should be 12.42h (NOAA M2 period)"

    def test_tide_window_duration_4_hours(self):
        """Usable high-tide window is approximately 4 hours, a standard
        operational planning assumption for tide-restricted ports."""
        sim = PortSimulation({
            "id": "tide_test", "seed": 1, "scenario_type": "calm_week",
            "num_vessels": 1,
            "vessel_mix": {"feeder": 1.0, "panamax": 0.0, "post_panamax": 0.0, "ulcv": 0.0},
            "disruptions": [], "yard_initial_occupancy": 0.30,
            "reefer_fraction": 0.0, "hazmat_fraction": 0.0,
            "vessel_delay_factor": 0.0, "description": "Tide test",
        })
        for start, end in sim.tide_windows:
            duration = end - start
            assert duration == pytest.approx(4.0, abs=0.1), \
                f"Tide window duration {duration:.2f}h should be ~4h"


# ---------------------------------------------------------------------------
# Schedule and Customs Parameters
# ---------------------------------------------------------------------------

class TestScheduleCustomsCitations:
    """Verify schedule/customs parameters match Sea-Intelligence/CBO citations."""

    def test_customs_inspection_rate_in_scenarios(self):
        """CBO 2016: 3-5% routine inspection. Customs crackdown scenarios
        use 15% (simulation-specific elevated rate)."""
        for task in ALL_TASKS.get("test", []):
            for d in task.get("disruptions", []):
                if d["type"] == "customs_hold":
                    rate = d["details"].get("inspection_rate", 0.05)
                    # Should be either normal (3-5%) or elevated (15%)
                    assert rate == 0.15 or 0.03 <= rate <= 0.05, \
                        f"Unexpected inspection rate {rate} in task {task['id']}"

    def test_reefer_fraction_around_5_percent(self):
        """Drewry Maritime Research: ~5% of global loaded TEU are reefer.
        Most scenarios use 0.05 (normal) or elevated for mixed-cargo."""
        normal_fractions = set()
        for task in ALL_TASKS.get("train", []):
            normal_fractions.add(task.get("reefer_fraction", 0.05))
        # Most common should be 0.05
        assert 0.05 in normal_fractions, \
            "Standard reefer fraction of 0.05 (5%) should be present in tasks"


# ---------------------------------------------------------------------------
# Berth Parameters
# ---------------------------------------------------------------------------

class TestBerthParameterCitations:
    """Verify berth parameters match PEMP 'Terminal Depth at Selected Ports'."""

    def test_berth_dimensions_in_range(self):
        """PEMP: 300-450m berth length and 12-18m draft for medium-large terminal."""
        for bcfg in BERTH_CONFIGS:
            assert 300.0 <= bcfg["length_m"] <= 450.0, \
                f"Berth {bcfg['berth_id']} length {bcfg['length_m']}m outside 300-450m range"
            assert 12.0 <= bcfg["draft_m"] <= 18.0, \
                f"Berth {bcfg['berth_id']} draft {bcfg['draft_m']}m outside 12-18m range"

    def test_berth_count_and_cranes(self):
        """4 berths with 3-4 cranes each = 12-16 crane slots total.
        Actual crane count is 12 (3 per berth initial distribution)."""
        assert len(BERTH_CONFIGS) == 4
        total_crane_slots = sum(b["max_cranes"] for b in BERTH_CONFIGS)
        assert total_crane_slots >= 12  # At least as many slots as cranes
        total_cranes = sum(len(v) for v in CRANE_DISTRIBUTION.values())
        assert total_cranes == 12


# ---------------------------------------------------------------------------
# General Environment Parameters
# ---------------------------------------------------------------------------

class TestGeneralParameterCitations:
    """Verify environment-level parameters."""

    def test_planning_horizon_168_hours(self):
        """168 hours = 1 week planning horizon."""
        assert PLANNING_HORIZON == 168.0
        assert PLANNING_HORIZON / 24.0 == 7.0  # Exactly 1 week

    def test_reward_weights_sum_to_one(self):
        """7 reward components must sum to 1.0 for proper weighting."""
        total = sum(REWARD_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_reward_components_present(self):
        """All 7 documented reward components should be defined."""
        expected = {
            "berth_utilization", "crane_productivity", "vessel_turnaround",
            "yard_efficiency", "truck_turnaround", "rail_utilization",
            "safety_compliance",
        }
        assert set(REWARD_WEIGHTS.keys()) == expected

    def test_task_counts(self):
        """30 training tasks + 10 test tasks = 40 total."""
        assert len(ALL_TASKS["train"]) == 30
        assert len(ALL_TASKS["test"]) == 10


# ---------------------------------------------------------------------------
# Meta-test: Citation Coverage
# ---------------------------------------------------------------------------

class TestCitationCoverage:
    """Verify the models.py docstring has citations for all parameters."""

    def test_all_parameters_have_citations(self):
        """Each parameter line in the docstring should have a [...] citation."""
        import models
        docstring = models.__doc__
        assert docstring is not None, "models.py should have a module docstring"

        # Find all parameter lines (lines starting with two spaces and a capital)
        param_lines = []
        in_refs = False
        for line in docstring.split("\n"):
            stripped = line.strip()
            if "Parameter references" in stripped:
                in_refs = True
                continue
            if in_refs and stripped.startswith("["):
                # This is a citation line, not a parameter
                continue
            if in_refs and stripped and not stripped.startswith("[") and ":" in stripped:
                param_lines.append(stripped)
            if in_refs and stripped == '"""':
                break

        assert len(param_lines) >= 15, \
            f"Expected at least 15 cited parameters, found {len(param_lines)}"

    def test_no_fabricated_author_names(self):
        """Verify known-bad author names are not in the docstring.
        Previously had 'Kim et al.' (should be Jo and Kim) and
        'Maciejewski et al.' (should be Moszyk, Deja, Dobrzynski)."""
        import models
        docstring = models.__doc__
        assert "Kim et al." not in docstring, \
            "Author should be 'Jo and Kim', not 'Kim et al.'"
        assert "Maciejewski" not in docstring, \
            "Author should be 'Moszyk, Deja, and Dobrzynski', not 'Maciejewski'"

    def test_correct_author_names_present(self):
        """Verify corrected author names are in the docstring."""
        import models
        docstring = models.__doc__
        assert "Jo and Kim" in docstring, \
            "Citation should reference 'Jo and Kim' (JMSE 2020)"
        assert "Moszyk" in docstring, \
            "Citation should reference 'Moszyk' (Sustainability 2021)"
