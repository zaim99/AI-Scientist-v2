import argparse
import json
import os.path as osp
import re
import traceback
from typing import Any, Dict, List

from ai_scientist.llm import (
    AVAILABLE_LLMS,
    create_client,
    get_response_from_llm,
)

from ai_scientist.tools.semantic_scholar import SemanticScholarSearchTool
from ai_scientist.tools.base_tool import BaseTool

# Create tool instances
semantic_scholar_tool = SemanticScholarSearchTool()

# Define tools at the top of the file
tools = [
    semantic_scholar_tool,
    {
        "name": "FinalizeIdea",
        "description": """Finalize your idea by providing the idea details.

The IDEA JSON should include the following fields:
- "Name": A short descriptor of the idea. Lowercase, no spaces, underscores allowed.
- "Title": A catchy and informative title for the proposal.
- "Short Hypothesis": A concise statement of the main hypothesis or research question. Clarify the need for this specific direction, ensure this is the best setting to investigate this idea, and there are not obvious other simpler ways to answer the question.
- "Related Work": A brief discussion of the most relevant related work and how the proposal clearly distinguishes from it, and is not a trivial extension.
- "Abstract": An abstract that summarizes the proposal in conference format (approximately 250 words).
- "Experiments": A list of experiments that would be conducted to validate the proposal. Ensure these are simple and feasible. Be specific in exactly how you would test the hypothesis, and detail precise algorithmic changes. Include the evaluation metrics you would use.
- "Risk Factors and Limitations": A list of potential risks and limitations of the proposal.""",
    },
]

# Create a tools dictionary for easy lookup
tools_dict = {tool.name: tool for tool in tools if isinstance(tool, BaseTool)}

# Create a string with the tool descriptions
tool_descriptions = "\n\n".join(
    (
        f"- **{tool.name}**: {tool.description}"
        if isinstance(tool, BaseTool)
        else f"- **{tool['name']}**: {tool['description']}"
    )
    for tool in tools
)

# Extract tool names for the prompt
tool_names = [
    f'"{tool.name}"' if isinstance(tool, BaseTool) else f'"{tool["name"]}"'
    for tool in tools
]
tool_names_str = ", ".join(tool_names)

system_prompt = f"""You are an experienced AI researcher who aims to propose high-impact research ideas resembling exciting grant proposals. Feel free to propose any novel ideas or experiments; make sure they are novel. Be very creative and think out of the box. Each proposal should stem from a simple and elegant question, observation, or hypothesis about the topic. For example, they could involve very interesting and simple interventions or investigations that explore new possibilities or challenge existing assumptions. Clearly clarify how the proposal distinguishes from the existing literature.

Ensure that the proposal can be done starting from the provided codebase, and does not require resources beyond what an academic lab could afford. These proposals should lead to papers that are publishable at top ML conferences.

You have access to the following tools:

{tool_descriptions}

Respond in the following format:

THOUGHT:
<Your thoughts here. Describe your reasoning and any analysis of the idea and the results returned from the tool. If you are finalizing your idea, state "I am done with the proposal.">

ACTION:
<The action to take, exactly one of {tool_names_str}>

ARGUMENTS:
<If ACTION is "SearchSemanticScholar", provide the search query as {{"query": "your search query"}}. If ACTION is "FinalizeIdea", provide the idea details as {{"idea": {{ ... }}}} with the IDEA JSON specified below.>

If you choose to finalize your idea, provide the IDEA JSON in the arguments:

IDEA JSON:
```json
{{
    "Name": "...",
    "Title": "...",
    "Short Hypothesis": "...",
    "Related Work": "...",
    "Abstract": "...",
    "Experiments": "...",
    "Risk Factors and Limitations": "..."
}}
```

Ensure the JSON is properly formatted for automatic parsing.

Note: You should perform at least one literature search before finalizing your idea to ensure it is well-informed by existing research."""

# Define the initial idea generation prompt
idea_generation_prompt = """{task_description}

<experiment.py>
{code}
</experiment.py>

Here are the proposals that you have already generated:

'''
{prev_ideas_string}
'''

Begin by generating an interestingly new high-level research proposal that differs from what you have previously proposed.
"""

# Define the reflection prompt
idea_reflection_prompt = """Round {current_round}/{num_reflections}.

In your thoughts, first carefully consider the quality, novelty, and feasibility of the proposal you just created.
Include any other factors that you think are important in evaluating the proposal.
Ensure the proposal is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try to refine and improve your proposal.
Stick to the spirit of the original idea unless there are glaring issues.

If you have new information from tools, such as literature search results, incorporate them into your reflection and refine your proposal accordingly.

Results from your last action (if any):

{last_tool_results}
"""


def generate_ideas(
    base_dir: str,
    client: Any,
    model: str,
    skip_generation: bool = False,
    max_num_generations: int = 20,
    num_reflections: int = 5,
) -> List[Dict]:
    if skip_generation:
        # Load existing ideas from file
        try:
            with open(osp.join(base_dir, "ideas.json"), "r") as f:
                ideas = json.load(f)
            print("Loaded existing proposals:")
            for idea in ideas:
                print(idea)
            return ideas
        except FileNotFoundError as e:
            print("No existing proposals found:")
            traceback.print_exc()
            print("Generating new proposals.")
        except json.JSONDecodeError as e:
            print("Error decoding existing proposals:")
            traceback.print_exc()
            print("Generating new proposals.")

    idea_str_archive = []

    with open(osp.join(base_dir, "experiment.py"), "r") as f:
        code = f.read()

    with open(osp.join(base_dir, "prompt.json"), "r") as f:
        prompt = json.load(f)

    task_description = prompt["task_description"]

    for gen_idx in range(max_num_generations):
        print()
        print(f"Generating proposal {gen_idx + 1}/{max_num_generations}")
        try:
            prev_ideas_string = "\n\n".join(idea_str_archive)

            last_tool_results = ""
            idea_finalized = False
            msg_history = []

            for reflection_round in range(num_reflections):
                if reflection_round == 0:
                    # Use the initial idea generation prompt
                    prompt_text = idea_generation_prompt.format(
                        task_description=task_description,
                        code=code,
                        prev_ideas_string=prev_ideas_string,
                    )
                else:
                    # Use the reflection prompt, including tool results if any
                    prompt_text = idea_reflection_prompt.format(
                        current_round=reflection_round + 1,
                        num_reflections=num_reflections,
                        last_tool_results=last_tool_results or "No new results.",
                    )

                response_text, msg_history = get_response_from_llm(
                    msg=prompt_text,
                    client=client,
                    model=model,
                    system_message=system_prompt,
                    msg_history=msg_history,
                )

                # Parse the LLM's response
                try:
                    # Use regular expressions to extract the components
                    thought_pattern = r"THOUGHT:\s*(.*?)\s*ACTION:"
                    action_pattern = r"ACTION:\s*(.*?)\s*ARGUMENTS:"
                    arguments_pattern = r"ARGUMENTS:\s*(.*?)(?:$|\nTHOUGHT:|\n$)"

                    thought_match = re.search(
                        thought_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )
                    action_match = re.search(
                        action_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )
                    arguments_match = re.search(
                        arguments_pattern, response_text, re.DOTALL | re.IGNORECASE
                    )

                    if not all([thought_match, action_match, arguments_match]):
                        raise ValueError("Failed to parse the LLM response.")

                    thought = thought_match.group(1).strip()
                    action = action_match.group(1).strip()
                    arguments_text = arguments_match.group(1).strip()

                    print(f"Thought:\n{thought}\n")
                    print(f"Action: {action}")
                    print(f"Arguments: {arguments_text}")

                    # Process the action and arguments
                    if action in tools_dict:
                        # It's a tool we have defined
                        tool = tools_dict[action]
                        # Parse arguments
                        try:
                            arguments_json = json.loads(arguments_text)
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid arguments JSON for {action}.")

                        # Use the tool
                        try:
                            # Assuming the arguments match the parameters of the tool
                            result = tool.use_tool(**arguments_json)
                            last_tool_results = result
                        except Exception as e:
                            last_tool_results = f"Error using tool {action}: {str(e)}"
                    elif action == "FinalizeIdea":
                        # Parse arguments
                        try:
                            arguments_json = json.loads(arguments_text)
                            idea = arguments_json.get("idea")
                            if not idea:
                                raise ValueError("Missing 'idea' in arguments.")

                            # Append the idea to the archive
                            idea_str_archive.append(json.dumps(idea))
                            print(f"Proposal finalized: {idea}")
                            idea_finalized = True
                            break
                        except json.JSONDecodeError:
                            raise ValueError("Invalid arguments JSON for FinalizeIdea.")
                    else:
                        print(
                            "Invalid action. Please specify one of the available tools."
                        )
                        print(f"Available actions are: {tool_names_str}")
                except Exception as e:
                    print(
                        f"Failed to parse LLM response. Response text:\n{response_text}"
                    )
                    traceback.print_exc()
                    break  # Exit the loop if parsing fails

            if idea_finalized:
                continue  # Move to the next idea

        except Exception as e:
            print("Failed to generate proposal:")
            traceback.print_exc()
            continue

    # Save ideas
    ideas = [json.loads(idea_str) for idea_str in idea_str_archive]

    with open(osp.join(base_dir, "ideas.json"), "w") as f:
        json.dump(ideas, f, indent=4)

    return ideas


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate AI scientist proposals")
    parser.add_argument(
        "--experiment",
        type=str,
        default="nanoGPT",
        help="Experiment to run AI Scientist on.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-05-13",
        choices=AVAILABLE_LLMS,
        help="Model to use for AI Scientist.",
    )
    parser.add_argument(
        "--skip-idea-generation",
        action="store_true",
        help="Skip proposal generation and use existing proposals.",
    )
    parser.add_argument(
        "--max-num-generations",
        type=int,
        default=50,
        help="Maximum number of proposal generations.",
    )
    parser.add_argument(
        "--num-reflections",
        type=int,
        default=5,
        help="Number of reflection rounds per proposal.",
    )
    args = parser.parse_args()

    # Create the LLM client
    client, client_model = create_client(args.model)

    base_dir = osp.join("templates", args.experiment)
    ideas = generate_ideas(
        base_dir=base_dir,
        client=client,
        model=client_model,
        skip_generation=args.skip_idea_generation,
        max_num_generations=args.max_num_generations,
        num_reflections=args.num_reflections,
    )
