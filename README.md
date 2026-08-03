# PortManager

[![⭐ OpenReward Environment](https://img.shields.io/badge/%E2%AD%90%20OpenReward-Environment-f7e6cc)](https://openreward.ai/GeneralReasoning/portmanager)

## Description

PortManager is a container port terminal management environment where agents schedule cranes, berths, yard storage, and truck/rail departures over a 168-hour (1-week) planning horizon. The simulation models a medium-large port with realistic vessel arrivals, STS crane operations, yard management, gate/rail logistics, and disruption events (storms, labor strikes, customs delays, equipment breakdowns).

Note: this is a synthetic environment that should be tested thoroughly before use in an RL pipeline.

## Capabilities

- Multi-step scheduling of vessels, cranes, yard storage, and transportation
- Managing disruption events (storms, labor strikes, equipment failures, customs holds)
- Balancing berth utilization, crane productivity, and yard efficiency
- Coordinating truck and rail departures
- Handling tide-dependent berthing for deep-draft vessels
- Enforcing safety constraints (hazmat segregation, reefer power requirements)

## Compute Requirements

No sandbox or special compute required. PortManager is a pure discrete-event simulation that runs in-process.

## License

[ORLv1](https://openreward.ai/orlv1.md).

## Tasks

There are 30 training tasks across 6 scenario types:

- **Calm Week** (5 tasks): Normal operations with minor disruptions
- **Storm Season** (5 tasks): Two major storms causing crane shutdowns
- **Labor Dispute** (5 tasks): Partial strike reduces productivity 50%
- **Peak Season** (5 tasks): 30% more vessel arrivals than normal
- **Equipment Aging** (5 tasks): Frequent crane breakdowns
- **Mixed Cargo** (5 tasks): High proportion of reefer and hazmat cargo

And 10 test tasks across 4 scenario types:

- **Perfect Storm** (3 tasks): Storm + labor dispute + vessel delays + equipment failure
- **Mega Vessel Week** (3 tasks): Multiple ULCV arrivals competing for deep-draft berths
- **Customs Crackdown** (2 tasks): Elevated 15% inspection rate blocking yard space
- **Cascade Crisis** (2 tasks): Multiple delayed vessels causing yard overflow

Each task simulates a 168-hour (1-week) planning horizon with 6-14 vessel arrivals.

## Reward Structure

This is a dense, verifiable reward environment. No LLM graders are used. Each
`advance_time` call returns a score for the hours it advanced through, and the
episode score combines two parts:

**Hourly components (55%).** Sampled once per *simulated hour* — not once per
`advance_time` call — so the score does not depend on how the agent slices time:

| component | weight | measures |
|---|---|---|
| berth_responsiveness | 0.25 | ships left waiting while a berth they could use sits empty |
| vessel_turnaround | 0.25 | ships worked within their target turnaround |
| crane_productivity | 0.20 | moves achieved against the work available |
| yard_efficiency | 0.10 | yard occupancy held in a 0.50-0.80 band |
| truck_turnaround | 0.10 | share of trucks served within 60 minutes |
| rail_utilization | 0.10 | fill rate of departed trains |

A component reads `n/a` for hours where it has nothing to measure (no ship in
port, no trucks moving); those hours are scored on the remaining components with
weights renormalised, so idle time is neither rewarded nor punished.
`safety_compliance` multiplies the hour's score rather than adding to it, and
overtime is charged for each hour it is worked.

Note that berth *responsiveness* is deliberately not berth *occupancy*: rewarding
occupancy penalised working a ship faster, because finishing it emptied the berth.

**Episode outcomes (45%).** Ships cleared, and ships cleared inside their target
turnaround. The hourly components are all time-averaged rates and compress
heavily, so serving one fewer ship out of ten barely moved them; these terms are
what make service actually delivered visible in the score.

**Step rewards are meant to be summed.** Only `advance_time` and `submit_plan`
carry one; other tools leave it unset rather than returning 0.0, so a stream of
zeros cannot dilute the reduction. Each `advance_time` banks the hourly share
for the hours it covered, and the terminal step banks the outcome share, so
nothing is counted twice. Two consequences:

- **Progress is rewarded.** An agent that only reaches hour 80 banks roughly
  half the hourly share, so getting further through the week is worth more.
  A run killed before it finishes still keeps what it banked.
- **Slicing time finely is worth nothing.** Because each interval banks the sum
  of its hours rather than their average, 168 one-hour advances and 14
  twelve-hour advances bank the same total for the same play.

The summed total approaches `total_reward` but falls slightly short of it: the
hourly components average over hours that had something to score, while the
banked credit divides by the full 168, so hours when the port sat empty bank
nothing.

## Tools

Agents have access to 10 tools:

1. **observe_port()** - Full snapshot of berths, cranes, yard, gates, rail, weather, disruptions
2. **assign_berth(vessel_id, berth_id)** - Dock a waiting vessel (validates draft, availability, tide)
3. **assign_cranes(vessel_id, crane_ids)** - Assign STS cranes to a berthed vessel
4. **move_crane(crane_id, berth_id)** - Relocate a crane between berths (1 hour)
5. **set_yard_plan(vessel_id, yard_block_ids, container_type)** - Designate yard blocks for containers
6. **dispatch_trucks(count, yard_block_id, gate_id)** - Route trucks between yard and gate
7. **schedule_train(track_id, yard_block_ids, departure_hour)** - Schedule a rail departure
8. **advance_time(hours)** - Advance simulation 1-12 hours (returns per-step reward)
9. **handle_disruption(disruption_id, action)** - Respond to active disruptions
10. **submit_plan()** - End episode and compute final reward

## Time Horizon

PortManager is an open-ended, long-horizon environment simulating a full week of port operations. A typical episode involves 40-60 advance_time calls plus observation and action calls.

## Other Environment Requirements

There are no further environment requirements; PortManager works out of the box with the OpenReward endpoint without any secrets.

## Safety

Agents in PortManager manage a simulated port terminal. The environment does not present direct safety risks as agents only interact with a synthetic simulation through scheduling decisions. No real-world systems, APIs, or data are involved.

The environment includes safety compliance as a reward component, teaching agents to respect IMDG hazmat segregation rules and reefer power requirements.

## Citations

```bibtex
@dataset{GRPortManager,
  author    = {General Reasoning Inc. Team},
  title     = {PortManager},
  year      = {2026},
  publisher = {OpenReward},
  url       = {https://openreward.ai/GeneralReasoning/portmanager}
}
```
