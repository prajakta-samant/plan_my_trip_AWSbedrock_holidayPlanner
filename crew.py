from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai_tools import SerperDevTool
import os
from crewai import LLM

from bedrock_agentcore import BedrockAgentCoreApp
app = BedrockAgentCoreApp()


serper_dev_tool = SerperDevTool(api_key="5606a34add07c257b14244d0c367fc867b97b00f")
llm=LLM(model="bedrock/us.amazon.nova-pro-v1:0")
@CrewBase
class PlanMyTrip():
    """PlanMyTrip crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def trip_researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['trip_researcher'], # type: ignore[index]
            verbose=True,
            tools=[serper_dev_tool],
            llm=llm
        )

    @agent
    def iternary_planner(self) -> Agent:
        return Agent(
            config=self.agents_config['iternary_planner'], # type: ignore[index]
            verbose=True,
            llm=llm
        )

    @task
    def research_task(self) -> Task:
        return Task(
            config=self.tasks_config['research_task'], # type: ignore[index]
        )

    @task
    def reporting_task(self) -> Task:
        return Task(
            config=self.tasks_config['reporting_task'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the PlanMyTrip crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
@app.entrypoint
def agent_invocation(payload, context):
    """Handler for agent invocation"""
    print(f'Payload: {payload}')
    try: 
        # Extract user input from payload
        user_input = payload.get("topic", "Tokyo, Japan")
        print(f"Processing vacation destination: {user_input}")
        
        # Crew Execution - Creates an instance of the VacationPlanner class and run crew method
        research_crew_instance = PlanMyTrip()
        crew = research_crew_instance.crew()
        # Starts the sequential agent workflow
        result = crew.kickoff(inputs={'topic': user_input})

        print("Context:\n-------\n", context)
        print("Result Raw:\n*******\n", result.raw)
        
        # Safely access json_dict if it exists
        if hasattr(result, 'json_dict'):
            print("Result JSON:\n*******\n", result.json_dict)
        
        return {"result": result.raw}
        
    except Exception as e:
        print(f'Exception occurred: {e}')
        return {"error": f"An error occurred: {str(e)}"}


# Local test function
def test_local():
    """Test the crew locally without AgentCore"""
    try:
        crew_instance = PlanMyTrip()
        crew = crew_instance.crew()
        result = crew.kickoff(inputs={'topic': 'Plan a vacation to Germany'})
        print("Result:", result.raw)
        return result
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    app.run(port=8080)  