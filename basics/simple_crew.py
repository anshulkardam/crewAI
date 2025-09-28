from dotenv import load_dotenv
from crewai import LLM
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
import os

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.2,
    api_key=API_KEY,
)

research_agent = Agent(
    role="Research Specialist",
    goal="Research interesting facts about the topic",
    backstory="You are an expert at finding relevant and factual data",
    verbose=True,
    llm=llm,
)

writer_agent = Agent(
    role="Creative Writer",
    goal="Write a short blog summary using the research",
    backstory="You are skilled at writing engaging summaries based on the provided content",
    llm=llm,
    verbose=True,
)

find_facts_task = Task(
    description="Find 3 to 5 interesting and recent facts about {topic}",
    expected_output="A bullet list of 3 to 5 facts",
    agent=research_agent,
)

write_summary_task = Task(
    description="Write a 500 word blog post summary about {topic} using the facts from the research agent",
    expected_output="A blog post summary",
    agent=writer_agent,
    context=[find_facts_task],
)

crew = Crew(
    agents=[research_agent, writer_agent],
    tasks=[find_facts_task, write_summary_task],
    verbose=True,
)

while True:

    query = input("Enter Topic for the Agent('q' for exit): ")

    if "q" in query:
        break

    crew.kickoff(inputs={"topic": query})
