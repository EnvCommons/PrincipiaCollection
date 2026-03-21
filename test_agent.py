import json
import asyncio
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward


MODEL_NAME = "gpt-5.2"
ENV_NAME = "GeneralReasoning/PrincipiaCollection"
SPLIT = "train"


async def main():
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

    environment = or_client.environments.get(name=ENV_NAME)
    tasks = await environment.list_tasks(split=SPLIT)
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")

    for task in tasks[:1]:
        rollout = or_client.rollout.create(
            run_name="principia_collection_test",
            rollout_name=f"test_{task.task_spec['id']}",
            environment=ENV_NAME,
            split=SPLIT,
            task_spec=task.task_spec,
        )

        async with environment.session(
            task=task,
            secrets={"openai_api_key": OPENAI_API_KEY},
        ) as session:
            prompt = await session.get_prompt()
            input_list = [{"role": "user", "content": prompt[0].text}]
            finished = False

            rollout.log_openai_response(message=input_list[0], is_finished=finished)

            while not finished:
                response = await oai_client.responses.create(
                    model=MODEL_NAME,
                    tools=tools,
                    input=input_list,
                )

                rollout.log_openai_response(response.output[-1])
                input_list += response.output

                for item in response.output:
                    if item.type == "function_call":
                        print(f"Tool: {item.name}")

                        tool_result = await session.call_tool(
                            item.name,
                            json.loads(str(item.arguments)),
                        )

                        reward = tool_result.reward
                        finished = tool_result.finished

                        input_list.append({
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": tool_result.blocks[0].text,
                        })

                        rollout.log_openai_response(
                            input_list[-1],
                            reward=reward,
                            is_finished=finished,
                        )

                        print(f"Reward: {reward:.4f} | Finished: {finished}")

                        if finished:
                            break


if __name__ == "__main__":
    asyncio.run(main())
