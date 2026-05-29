import os
from typing import TypedDict, List, Dict, Any
import questionary
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Define the state of the graph
class GameState(TypedDict):
    theme: str
    scene: str
    options: List[str]
    choice: str | None
    ending: str

# LLM client (OpenAI)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))

# Prompt for generating scene and options
scene_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative storyteller.") ,
    ("user", "Generate a short opening (2-3 sentences) about the theme: {theme}. Then provide exactly three actions the hero can take. Respond with the scene first, then a numbered list of options on separate lines."),
])

# Prompt for ending based on choice
ending_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise storyteller.") ,
    ("user", "Given the opening: {scene}\nUser chose option: {choice}\nWrite a short ending (2-3 sentences)."),
])

# Node to generate scene and interrupt for choice
async def generate_scene(state: GameState) -> Dict[str, Any]:
    theme = state["theme"]
    # Call LLM
    result = await llm.ainvoke(scene_prompt.format(theme=theme))
    text = result.content.strip()
    # Split scene and options
    parts = text.split("\n")
    scene_text = parts[0]
    opts = [p.strip() for p in parts[1:4]]
    state.update({"scene": scene_text, "options": opts})
    # Prepare interrupt payload
    payload = {
        "type": "choice",
        "question": f"{scene_text}\nWhat do you do?",
        "choices": opts,
    }
    return interrupt(payload)

# Node to process choice and generate ending
async def finish_story(state: GameState) -> Dict[str, Any]:
    # state now contains 'choice'
    scene = state["scene"]
    choice = state["choice"]
    result = await llm.ainvoke(ending_prompt.format(scene=scene, choice=choice))
    ending_text = result.content.strip()
    state.update({"ending": ending_text})
    return state

# Build graph
graph_builder = StateGraph(GameState)
graph_builder.add_node("generate", generate_scene)
graph_builder.add_node("finish", finish_story)
graph_builder.set_entry_point("generate")
graph_builder.add_edge(START, "generate")
graph_builder.add_edge("generate", "finish")
# No further edges; graph ends after finish
graph = graph_builder.compile(checkpointer=InMemorySaver())

# Run the graph with interrupt handling
async def run_game(theme: str):
    from langchain_core.messages import BaseMessage
    config = {"configurable": {"thread_id": "demo_thread"}}
    # Start streaming
    async for chunk in graph.stream(Command(), config=config):
        if isinstance(chunk, dict) and "__interrupt__" in chunk:
            interrupt_payload = chunk["__interrupt__"][0].value
            answer = questionary.select(
                interrupt_payload["question"],
                choices=interrupt_payload["choices"],
            ).ask()
            # Resume with user's choice
            resume_payload = {**interrupt_payload, "choice": answer}
            async for _ in graph.stream(Command(resume=rescue_payload), config=config):
                pass  # consume remaining chunks
        else:
            # Print normal messages (scene or ending)
            if isinstance(chunk, BaseMessage):
                print(chunk.content)
    # After completion, fetch final state
    final_state = graph.get_state(config).data
    print("\n--- Final State ---")
    print(final_state)

if __name__ == "__main__":
    import asyncio
    theme_input = questionary.text("Enter a theme for the adventure:").ask()
    asyncio.run(run_game(theme_input))
