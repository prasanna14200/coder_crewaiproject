import os

from crewai import Agent, Crew, LLM, Process, Task
from crewai.project import CrewBase, agent, crew, task



@CrewBase
class Coder():
    """Coder crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    def __init__(self, *, allow_code_execution: bool = True):
        self.allow_code_execution = allow_code_execution

    def _llm(self) -> LLM:
        api_key = os.getenv('GEMINI_API_KEY', '').strip()
        if not api_key:
            raise RuntimeError('GEMINI_API_KEY is not configured.')
        return LLM(
            model=self.agents_config['coder']['llm'],
            api_key=api_key,
            timeout=30,
            max_tokens=1024,
            num_retries=0,
        )

    # One click install for Docker Desktop:
    #https://docs.docker.com/desktop/

    @agent
    def coder(self) -> Agent:
        return Agent(
            config=self.agents_config['coder'],
            llm=self._llm(),
            verbose=True,
            allow_code_execution=self.allow_code_execution,
            code_execution_mode="safe",  # Uses Docker for safety
            max_execution_time=60,
            max_retry_limit=0,
            allow_dangerous_code=False,
    )


    @task
    def coding_task(self) -> Task:
        return Task(
            config=self.tasks_config['coding_task'],
        )


    @crew
    def crew(self) -> Crew:
        """Creates the Coder crew"""


        return Crew(
            agents=self.agents, 
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
