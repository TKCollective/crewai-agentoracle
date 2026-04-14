# crewai-agentoracle

CrewAI tool for [AgentOracle](https://agentoracle.co) — trust verification for AI agents.

## Installation

```bash
pip install crewai-agentoracle
```

## Usage

```python
from crewai import Agent, Task, Crew
from crewai_agentoracle import AgentOracleVerifyTool, AgentOracleEvaluateTool

# Create tools
verify_tool = AgentOracleVerifyTool()
evaluate_tool = AgentOracleEvaluateTool()

# Create an agent with verification capabilities
researcher = Agent(
    role="Research Analyst",
    goal="Find and verify information before making recommendations",
    backstory="An analyst who always fact-checks before reporting.",
    tools=[verify_tool, evaluate_tool],
)

# The agent will automatically use AgentOracle to verify claims
task = Task(
    description="Research whether Bitcoin reached $100K in 2025 and verify the claim.",
    expected_output="Verified research finding with confidence score.",
    agent=researcher,
)

crew = Crew(agents=[researcher], tasks=[task])
result = crew.kickoff()
```

## Tools

| Tool | Description | Price |
|------|-------------|-------|
| `AgentOracleVerifyTool` | Quick claim verification via /preview | Free (10/hr) |
| `AgentOracleEvaluateTool` | Full 4-source verification | $0.02 USDC via x402 |

## License

MIT
