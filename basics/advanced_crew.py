from dotenv import load_dotenv
import os
from crewai import LLM, Agent, Task, Crew
from crewai.tools import tool

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.1,
    api_key=API_KEY,
)


@tool("web_search")
def web_search_tool(query: str) -> str:
    """this tool gives result for the user query"""
    return (
        f"Results for '{query}':\n"
        "- UNESCO recommends ethical AI guidelines in education (2024).\n"
        "- India announced new AI curriculum for schools (2025).\n"
        "- OECD: AI literacy is a key 21st-century skill.\n"
    )


manager = Agent(
    role="Manager",
    goal="Plan the workflow and manage sub-agents",
    backstory="A manager that assigns subtasks, checks outputs, and retries if needed",
    llm=llm,
    tools=[],
)

worker_global = Agent(
    role="Worker-GlobalRegulations",
    goal="Execute the subtask: GlobalRegulations",
    backstory="A worker agent focused on global AI education policy research",
    llm=llm,
    tools=[web_search_tool],
)

worker_india = Agent(
    role="Worker-IndianContext",
    goal="Execute the subtask: IndianContext",
    backstory="A worker agent focused on India-specific AI education policy research",
    llm=llm,
    tools=[web_search_tool],
)

worker_recommend = Agent(
    role="Worker-Recommendations",
    goal="Execute the subtask: Recommendations",
    backstory="A worker agent focused on synthesizing recommendations",
    llm=llm,
    tools=[],
)

synth_worker = Agent(
    role="Synthesize",
    goal="Combine all outputs into a coherent policy brief",
    backstory="A synthesis agent that assembles and checks the final brief",
    llm=llm,
    tools=[],
)

main_prompt = "Produce a policy brief on regulating AI in education, with global and Indian context"

# Tasks
plan_task = Task(
    description=(
        f"You are the Manager. Plan 3 subtasks (titles + descriptions) to achieve: {main_prompt}. "
        "Return a list of subtask names and descriptions."
    ),
    agent=manager,
    expected_output="A list of 3 subtask names and their descriptions.",
)

worker_global_task = Task(
    description=(
        "Based on the Manager’s plan, execute subtask: GlobalRegulations. "
        "Research global AI education policies using web_search and write ~2 paragraphs."
    ),
    agent=worker_global,
    expected_output="2 paragraphs on global AI education policies.",
    context=[plan_task],
)

worker_india_task = Task(
    description=(
        "Execute subtask: IndianContext. "
        "Research India-specific AI regulation in education using web_search and write ~2 paragraphs."
    ),
    agent=worker_india,
    expected_output="2 paragraphs on India's AI education policy.",
    context=[plan_task],
)

worker_recommend_task = Task(
    description=(
        "Execute subtask: Recommendations. "
        "Based on other two outputs, propose 3 actionable recommendations in 2 paragraphs."
    ),
    agent=worker_recommend,
    expected_output="2 paragraphs with 3 actionable recommendations.",
    context=[worker_global_task, worker_india_task],
)

synth_task = Task(
    description=(
        "You are the Synthesize worker. Combine the outputs from the 3 subtasks "
        "into a coherent policy brief (~4–5 paragraphs). Check consistency."
    ),
    agent=synth_worker,
    expected_output="A final policy brief (4–5 paragraphs).",
    context=[worker_global_task, worker_india_task, worker_recommend_task],
)

crew = Crew(
    agents=[manager, worker_global, worker_india, worker_recommend, synth_worker],
    tasks=[
        plan_task,
        worker_global_task,
        worker_india_task,
        worker_recommend_task,
        synth_task,
    ],
    verbose=True,
)

if __name__ == "__main__":
    result = crew.kickoff()
    print(result)
