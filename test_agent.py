import json
import asyncio
import os
from pathlib import Path

import nest_asyncio
nest_asyncio.apply()

from openai import AsyncOpenAI
from openreward import OpenReward


async def main():
    or_client = OpenReward()
    oai_client = AsyncOpenAI()

    MODEL_NAME = "gpt-5.2"
    ENV_NAME = "GeneralReasoning/portmanager"
    SPLIT = "train"
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    environment = or_client.environments.get(name=ENV_NAME, base_url="http://localhost:8081")
    tasks = environment.list_tasks(split=SPLIT)
    tools = environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")
    tool_names = [t.get('name') or t.get('function', {}).get('name', '?') for t in tools]
    print(f"Available tools: {tool_names}")

    trajectory_path = Path("trajectory.jsonl")

    for task in tasks[:1]:
        print(f"\n{'='*60}")
        print(f"Starting task: {task.task_spec.get('id', 'unknown')}")
        print(f"Scenario: {task.task_spec.get('scenario_type', 'unknown')}")
        print(f"Vessels: {task.task_spec.get('num_vessels', '?')}")
        print(f"{'='*60}")

        rollout = or_client.rollout.create(
            run_name=ENV_NAME.split("/")[-1] + "_test",
            rollout_name="test_run",
            environment=ENV_NAME,
            split=SPLIT,
            task_spec=task.task_spec,
        )

        with open(trajectory_path, "w") as traj_file:
            with environment.session(task=task, secrets={"openai_api_key": OPENAI_API_KEY}) as session:
                prompt = session.get_prompt()
                input_list = [{"role": "user", "content": prompt[0].text}]
                finished = False
                turn = 0
                cumulative_reward = 0.0

                # Log initial prompt
                traj_entry = {
                    "turn": turn,
                    "clock": 0.0,
                    "action": "prompt",
                    "tool_call": None,
                    "result_preview": prompt[0].text[:500],
                    "reward": None,
                    "cumulative_reward": 0.0,
                    "finished": False,
                }
                traj_file.write(json.dumps(traj_entry) + "\n")
                traj_file.flush()

                rollout.log_openai_response(message=input_list[0], is_finished=finished)

                while not finished:
                    turn += 1
                    response = await oai_client.responses.create(
                        model=MODEL_NAME,
                        tools=tools,
                        input=input_list,
                    )

                    rollout.log_openai_response(response.output[-1])
                    input_list += response.output

                    for item in response.output:
                        if item.type == "function_call":
                            tool_result = session.call_tool(
                                item.name, json.loads(str(item.arguments))
                            )

                            reward = tool_result.reward
                            finished = tool_result.finished
                            cumulative_reward += reward

                            input_list.append({
                                "type": "function_call_output",
                                "call_id": item.call_id,
                                "output": tool_result.blocks[0].text,
                            })
                            rollout.log_openai_response(
                                input_list[-1], reward=reward, is_finished=finished
                            )

                            # Log trajectory entry
                            traj_entry = {
                                "turn": turn,
                                "clock": tool_result.metadata.get("clock", None)
                                         if isinstance(tool_result.metadata, dict) else None,
                                "action": item.name,
                                "tool_call": json.loads(str(item.arguments)),
                                "result_preview": tool_result.blocks[0].text[:500],
                                "reward": reward,
                                "cumulative_reward": round(cumulative_reward, 4),
                                "finished": finished,
                            }

                            # Add step reward breakdown if present
                            if isinstance(tool_result.metadata, dict):
                                step_reward = tool_result.metadata.get("step_reward")
                                if step_reward:
                                    traj_entry["step_reward"] = step_reward

                                # Add final reward breakdown
                                if "total_reward" in tool_result.metadata:
                                    traj_entry["reward_breakdown"] = {
                                        k: tool_result.metadata[k]
                                        for k in [
                                            "total_reward", "num_steps",
                                            "vessels_completed", "vessels_total",
                                            "total_crane_moves", "trains_departed",
                                            "safety_violations",
                                            "avg_berth_utilization",
                                            "avg_crane_productivity",
                                            "avg_vessel_turnaround",
                                            "avg_yard_efficiency",
                                            "avg_truck_turnaround",
                                            "avg_rail_utilization",
                                            "avg_safety_compliance",
                                        ]
                                        if k in tool_result.metadata
                                    }

                            traj_file.write(json.dumps(traj_entry) + "\n")
                            traj_file.flush()

                            print(f"Turn {turn:3d} | {item.name:20s} | "
                                  f"Reward: {reward:.4f} | "
                                  f"Cumul: {cumulative_reward:.4f} | "
                                  f"Finished: {finished}")

                            if finished:
                                print("\nFINISHED!")
                                if isinstance(tool_result.metadata, dict) and "total_reward" in tool_result.metadata:
                                    md = tool_result.metadata
                                    print(f"\n{'='*55}")
                                    print(f"FINAL RESULTS")
                                    print(f"{'='*55}")
                                    print(f"  Total Reward:      {md['total_reward']:.4f}")
                                    print(f"  Vessels completed: {md.get('vessels_completed')}/{md.get('vessels_total')}")
                                    print(f"  Total crane moves: {md.get('total_crane_moves')}")
                                    print(f"  Trains departed:   {md.get('trains_departed')}")
                                    print(f"  Safety violations: {md.get('safety_violations')}")
                                    print(f"  Num steps:         {md.get('num_steps')}")
                                    print(f"")
                                    print(f"  Component Averages:")
                                    for comp in ["berth_utilization", "crane_productivity",
                                                  "vessel_turnaround", "yard_efficiency",
                                                  "truck_turnaround", "rail_utilization",
                                                  "safety_compliance"]:
                                        avg_key = f"avg_{comp}"
                                        print(f"    {comp:25s}: {md.get(avg_key, 'N/A')}")
                                    print(f"{'='*55}")
                                break

        print(f"\nTrajectory written to {trajectory_path}")
        print(f"Total turns: {turn}")
        print(f"Cumulative reward: {cumulative_reward:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
