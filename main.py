import os
from typing import TypedDict, List, Dict, Any
import questionary
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# -----------------------------
# 1. Define the graph state
# -----------------------------
class GameState(TypedDict):
    theme: str
    intro: str = ""
    options: List[str] = []
    choice: str | None = None
    ending: str = ""

# -----------------------------
# 2. LLM wrapper
# -----------------------------
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)

# Prompt for generating intro and options
intro_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative storyteller.") ,
    ("human", "Topic: {theme}\nGenerate a short introduction (2-3 sentences) followed by exactly three distinct actions the hero can take. Return the intro first, then list the options each on a new line.")
])

# Prompt for generating ending based on choice
ending_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a creative storyteller."),
    ("human", "Intro: {intro}\nChoice: {choice}\nWrite a short ending (2-3 sentences) that reflects the chosen action.")
])

# -----------------------------
# 3. Node to generate intro and options
# -----------------------------
async def generate_intro(state: GameState) -> Dict[str, Any]:
    theme = state["theme"]
    response = await llm.invoke(intro_prompt.format(theme=theme))
    text = response.content.strip()
    # Split into intro and options
    parts = text.split("\n")
    intro = parts[0]
    opts = [p for p in parts[1:] if p.strip()]
    state["intro"] = intro
    state["options"] = opts
    # Prepare interrupt payload
    payload = {
        "type": "choice",
        "question": f"{intro}\nWhat do you do?",
        "choices": opts,
    }
    return {"__interrupt__": [payload]}

# -----------------------------
# 4. Node to handle user choice and generate ending
# -----------------------------
async def process_choice(state: GameState) -> Dict[str, Any]:
    # The interrupt payload will be merged into state by the graph
    if state.get("choice") is None:
        raise ValueError("Choice not found in state after interruption.")
    response = await llm.invoke(ending_prompt.format(intro=state["intro"], choice=state["choice"]))
    ending = response.content.strip()
    state["ending"] = ending
    return state

# -----------------------------
# 5. Build the graph
# -----------------------------
graph_builder = StateGraph(GameState)
graph_builder.add_node("intro", generate_intro)
graph_builder.add_node("choice", process_choice)
graph_builder.set_entry_point("intro")
graph_builder.add_edge(START, "intro")
graph_builder.add_edge("intro", "choice")
# No further edges; graph ends after choice node
graph = graph_builder.compile(checkpointer=InMemorySaver())

# -----------------------------
# 6. Run the graph with user interaction loop
# -----------------------------
async def run_game(theme: str):
    config = {"configurable": {"thread_id": "demo-thread"}}
    # Start streaming; we will handle interrupts manually
    async for chunk in graph.stream(Command(), config):
        if isinstance(chunk, dict) and "__interrupt__" in chunk:
            payload = chunk["__interrupt__"][0]
            answer = questionary.select(
                message=payload["question"],
                choices=payload["choices"]
            ).ask()
            # Resume with user's choice
            resume_payload = {**payload, "choice": answer}
            async for _ in graph.stream(Command(resume=resume_payload), config):
                pass  # consume remaining chunks (ending)
        else:
            # Print normal text chunks
            print(chunk)
    # After completion, print final state
    final_state = await graph.get_state(config["configurable"]["thread_id"])
    print("\n--- Final State ---")
    print(f"Intro: {final_state['intro']}")
    print(f"Choice: {final_state['choice']}")
    print(f"Ending: {final_state['ending']}")

# -----------------------------
# 7. Entry point
# -----------------------------
if __name__ == "__main__":
    theme = questionary.text("Enter a theme for the story:").ask()
    import asyncio
    asyncio.run(run_game(theme))
