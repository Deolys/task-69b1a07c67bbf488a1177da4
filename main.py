# main.py
# Interactive choose-your-own-adventure using LangGraph, OpenAI LLM and questionary for console input.
# Requires Python 3.10+ and the following packages:
#   langgraph==0.1.0
#   langchain==1.2.10
#   langchain-openai==1.1.9
#   questionary
#
# The graph has two nodes: "generate_scene" and "write_end".
# It uses an interrupt to pause after the scene is generated, present options to the user,
# then resume with the chosen option to generate a short ending.

from __future__ import annotations

import os
import re
from typing import TypedDict, List, Dict, Any

import questionary
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command

# ---------------------------------------------------------------------------
# 1. Define the graph state
# ---------------------------------------------------------------------------
class AdventureState(TypedDict):
    theme: str | None
    scene: str | None
    options: List[str] | None
    choice: str | None
    ending: str | None

# ---------------------------------------------------------------------------
# 2. LLM wrapper
# ---------------------------------------------------------------------------
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

# Helper to call the model with a prompt and return raw text
def ask(prompt: str) -> str:
    """Send a prompt to the LLM and return the assistant's content."""
    response = llm.invoke([{"role": "user", "content": prompt}])
    return response.content.strip()

# ---------------------------------------------------------------------------
# 3. Node: generate_scene
# ---------------------------------------------------------------------------
async def generate_scene(state: AdventureState) -> Dict[str, Any]:
    theme = state["theme"] or "unknown"
    # Prompt to get scene and options
    prompt = (
        f"Theme: {theme}.\n"
        "Write a short opening (2-3 sentences). Then list exactly three actions the hero can take. "
        "Respond in this format:\n"
        "SCENE:\n<scene text>\nOPTIONS:\n1) <option 1>\n2) <option 2>\n3) <option 3>"
    )
    raw = ask(prompt)

    # Parse scene and options
    scene_match = re.search(r"SCENE:\s*(.*?)\s*OPTIONS:", raw, re.S)
    options_match = re.findall(r"\d+\)\s*(.+)", raw)
    scene_text = scene_match.group(1).strip() if scene_match else ""
    opts: List[str] = [o.strip() for o in options_match][:3]

    state["scene"] = scene_text
    state["options"] = opts

    # Prepare interrupt payload
    interrupt_payload = {
        "type": "choice",
        "question": f"{scene_text}\nWhat do you do?",
        "choices": opts,
    }
    return {"__interrupt__": [interrupt_payload]}

# ---------------------------------------------------------------------------
# 4. Node: write_end (resumes after interrupt)
# ---------------------------------------------------------------------------
async def write_end(state: AdventureState) -> AdventureState:
    # The state will contain the user's choice under "choice"
    choice = state.get("choice") or ""
    scene = state.get("scene") or ""
    theme = state.get("theme") or ""

    prompt = (
        f"Theme: {theme}.\nScene: {scene}\nUser chose: {choice}.\nWrite a short ending (2-3 sentences) that follows from this choice."
    )
    ending_text = ask(prompt)
    state["ending"] = ending_text
    return state

# ---------------------------------------------------------------------------
# 5. Build the graph
# ---------------------------------------------------------------------------
graph_builder = StateGraph(AdventureState, checkpoint=InMemorySaver())
graph_builder.add_node("generate_scene", generate_scene)
graph_builder.add_node("write_end", write_end)
graph_builder.set_entry_point("generate_scene")
# After interrupt, resume to write_end
graph_builder.add_edge("generate_scene", "write_end")
graph = graph_builder.compile()

# ---------------------------------------------------------------------------
# 6. Run the graph with user interaction loop
# ---------------------------------------------------------------------------
async def run_adventure(theme: str) -> None:
    config = {"configurable": {"thread_id": "adventure_thread"}}
    # Initial state
    init_state: AdventureState = {
        "theme": theme,
        "scene": None,
        "options": None,
        "choice": None,
        "ending": None,
    }
    stream = graph.stream(init_state, config)
    async for chunk in stream:
        if "__interrupt__" in chunk:
            # Extract payload
            interrupt_obj = chunk["__interrupt__"][0]
            question = interrupt_obj.get("question", "Choose an option:")
            choices: List[str] = interrupt_obj.get("choices", [])
            # Show menu to user
            answer = questionary.select(question, choices=choices).ask()
            # Resume with user's choice
            resume_payload = {"choice": answer}
            stream = graph.stream(Command(resume=rescue_payload), config)
        else:
            # Print normal output (scene or ending)
            if chunk.get("scene"):
                print(chunk["scene"])
            if chunk.get("ending"):
                print(chunk["ending"])

# ---------------------------------------------------------------------------
# 7. Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    theme_input = os.getenv("ADVENTURE_THEME", "space cat")
    import asyncio
    asyncio.run(run_adventure(theme_input))
