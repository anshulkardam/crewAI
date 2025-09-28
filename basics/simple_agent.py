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


email_assistant = Agent(
    role="Email Assistant",
    goal="Improve emails and make them sound professional and clear",
    backstory="A highly experienced communication expert skilled in professional email writing",
    verbose=True,
    llm=llm,
)

original_email = """
hey team, just wanted to tell you we are going into a sprint from monday, but there is some stuff left that we need to figure out, lets get that ready before
moving and be ready for the upc. sprint. thanks & all the best"""

email_task = Task(
    description=f"""Take the following rough email and rewrite it into professional and polished version.
    Expand abbreviations:
    '''{original_email}'''
    """,
    agent=email_assistant,
    expected_output="A professional written email with proper formatting and content",
)


crew = Crew(
    agents=[email_assistant],
    tasks=[email_task],
    verbose=True,
)

result = crew.kickoff()

print(result)
