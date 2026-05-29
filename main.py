import os
from typing import TypedDict, List, Dict, Any
import questionary
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt, Command

# Define the state of the graph
class GameState(TypedDict):
    theme: str
    intro: str
    options: List[str]
    choice: str | None
    ending: str

# LLM client (replace with your own key if needed)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7, api_key=os.getenv("OPENAI_API_KEY"))

# Helper to parse LLM output into intro and options
def parse_intro_and_options(text: str) -> tuple[str, List[str]]:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return "", []
    intro = lines[0]
    opts = []
    for line in lines[1:]:
        parts = line.split(")", 1)
        if len(parts) == 2:
            opt = parts[1].strip()
        else:
            opt = line.strip()
        opts.append(opt)
    return intro, opts

# Node: generate scene and interrupt for choice
async def scene_node(state: GameState) -> Dict[str, Any]:
    theme = state.get("theme", "unknown")
    prompt = (
        f"Тема: {theme}.\n"
        "Придумай короткую завязку (2–3 предложения) и ровно 3 варианта поступка героя.\n"
        "Ответь в формате:\n"
        "1) ...\n"
        "2) ...\n"
        "3) ..."
    )
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, 'content') else str(response)
    intro, opts = parse_intro_and_options(text)
    state["intro"] = intro
    state["options"] = opts
    payload = {
        "type": "choice",
        "question": f"{intro}\n\nЧто делаем?",
        "choices": opts,
    }
    return interrupt(payload)

# Node: finish with ending based on choice
async def ending_node(state: GameState) -> GameState:
    theme = state.get("theme", "unknown")
    intro = state.get("intro", "")
    choice = state.get("choice", "")
    prompt = (
        f"Тема: {theme}.\n"
        f"Завязка: {intro}.\n"
        f"Выбор пользователя: {choice}.\n"
        "Допиши короткую концовку (2–3 предложения)."
    )
    response = llm.invoke(prompt)
    ending = response.content if hasattr(response, 'content') else str(response)
    state["ending"] = ending
    return state

# Build the graph
graph_builder = StateGraph(GameState)
graph_builder.add_node("scene", scene_node)
graph_builder.add_node("ending", ending_node)
graph_builder.set_entry_point("scene")
graph_builder.add_edge(START, "scene")
graph_builder.add_edge("scene", "ending")
checkpoint = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpoint)

# Run the interactive loop
def run_game(theme: str):
    config = {"configurable": {"thread_id": theme}}
    # First stream to get interrupt
    for chunk in graph.stream(Command(), config=config):
        if "__interrupt__" in chunk:
            payload = chunk["__interrupt__"][0].value
            answer = questionary.select(
                message=payload["question"], choices=payload["choices"]
            ).ask()
            # Resume with user's choice
            resume_payload = {**payload, "choice": answer}
            for subchunk in graph.stream(Command(resume=rescue_payload), config=config):
                if "__interrupt__" not in subchunk:
                    print(subchunk.get("ending", ""))
            break
    else:
        pass

if __name__ == "__main__":
    theme = questionary.text("Введите тему истории: ").ask()
    run_game(theme)
