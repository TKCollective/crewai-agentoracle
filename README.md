# crewai-agentoracle

**Trust verification for CrewAI agents. Per-claim. Before they act.**

CrewAI agents are only as good as what they believe. One hallucinated fact 
passed between agents corrupts the entire crew's output.

AgentOracle adds a verification step between what your agent retrieves 
and what it acts on. Every claim gets a verdict. Every verdict has a score.

---

## Install

```bash
pip install crewai-agentoracle
```

---

## Feedback & Support

Built something with AgentOracle? Hit an issue? We want to hear from you.

- **GitHub Discussions** — questions, ideas, show and tell: [github.com/TKCollective/x402-research-skill/discussions](https://github.com/TKCollective/x402-research-skill/discussions)
- **X / Twitter** — [@AgentOracle_AI](https://x.com/AgentOracle_AI)
- **Issues** — bugs and feature requests: open an issue in this repo

If you're evaluating AgentOracle for a project, drop a note in Discussions — we respond fast and can help with integration.

---

## Quickstart

```python
from crewai import Agent, Task, Crew
from crewai_agentoracle import AgentOracleVerifyTool

verify_tool = AgentOracleVerifyTool()

researcher = Agent(
    role="Research Analyst",
    goal="Research AI agent frameworks and verify all claims before reporting",
    backstory="You are a meticulous analyst who never reports unverified information.",
    tools=[verify_tool],
    verbose=True
)

task = Task(
    description="Research the top AI agent frameworks in 2026. Verify every major claim before including it in your report.",
    expected_output="A verified report on AI agent frameworks with confidence scores for each claim.",
    agent=researcher
)

crew = Crew(agents=[researcher], tasks=[task], verbose=True)
result = crew.kickoff()
```

That's it. Your crew now verifies before acting.

---

## What comes back

```json
{
  "overall_confidence": 0.91,
  "recommendation": "act",
  "claims": [
    {
      "claim": "LangGraph leads agent frameworks in 2026",
      "verdict": "supported",
      "confidence": 0.94,
      "evidence": "Confirmed across 4 independent sources"
    },
    {
      "claim": "CrewAI was acquired by Google in 2025",
      "verdict": "refuted",
      "confidence": 0.02,
      "correction": "CrewAI remains independent as of April 2026"
    }
  ]
}
```

---

## Recommendation logic

| Score | Recommendation | What your agent should do |
|-------|---------------|--------------------------|
| > 0.8 | `act` | Proceed — claims verified |
| 0.5–0.8 | `verify` | Pause — route to human review |
| < 0.5 | `reject` | Discard — evidence contradicted |

---

## How it works

Every evaluation runs through 4 independent sources in parallel:

1. **Sonar** — real-time web research
2. **Sonar Pro** — deep multi-step analysis
3. **Adversarial** — actively tries to disprove the claim
4. **Gemma 4** — claim decomposition and confidence calibration

Consensus builds the score. Contradiction flags the risk.

---

## Multi-agent crew example

```python
from crewai import Agent, Task, Crew
from crewai_agentoracle import AgentOracleVerifyTool

verify_tool = AgentOracleVerifyTool()

# Researcher finds the data
researcher = Agent(
    role="Research Analyst",
    goal="Find the latest information on AI agent frameworks",
    backstory="Expert researcher with access to real-time web data.",
    verbose=True
)

# Verifier checks it before the writer uses it
verifier = Agent(
    role="Fact Verifier",
    goal="Verify every claim the researcher finds before it goes to the writer",
    backstory="You trust nothing until it is verified. You use AgentOracle to score every claim.",
    tools=[verify_tool],
    verbose=True
)

# Writer only works with verified data
writer = Agent(
    role="Content Writer",
    goal="Write a report using only verified, high-confidence information",
    backstory="You write clearly and only include claims with confidence above 0.8.",
    verbose=True
)

research_task = Task(
    description="Research the top 5 AI agent frameworks in 2026.",
    expected_output="A list of frameworks with key claims about each.",
    agent=researcher
)

verify_task = Task(
    description="Verify every claim from the research. Flag anything below 0.8 confidence.",
    expected_output="Verified claims with confidence scores and verdicts.",
    agent=verifier,
    context=[research_task]
)

write_task = Task(
    description="Write a report using only the verified claims with confidence above 0.8.",
    expected_output="A clean, accurate report on AI agent frameworks.",
    agent=writer,
    context=[verify_task]
)

crew = Crew(
    agents=[researcher, verifier, writer],
    tasks=[research_task, verify_task, write_task],
    verbose=True
)

result = crew.kickoff()
print(result)
```

---

## Try it free — no setup needed

```bash
curl -X POST https://agentoracle.co/preview \
  -H "Content-Type: application/json" \
  -d '{"query": "CrewAI was acquired by Google in 2025"}'
```

20 free requests per hour. No wallet, no API key, no account.

---

## Pricing

| Endpoint | Price | What it does |
|----------|-------|-------------|
| `/preview` | Free | Truncated results, no payment needed |
| `/evaluate` | $0.01/claim | Full per-claim verification + verdicts |
| `/research` | $0.02/query | Real-time research + verification |

Payments via [x402 protocol](https://x402.org) — USDC on Base, SKALE (gasless), 
or Stellar. No subscriptions. No minimums. No API keys.

---

## Related

- [agentoracle.co](https://agentoracle.co) — main site + live demo
- [Trust Layer docs](https://agentoracle.co/trust) — full API reference
- [langchain-agentoracle](https://github.com/TKCollective/langchain-agentoracle) — LangChain integration
- [agentoracle-mcp](https://github.com/TKCollective/agentoracle-mcp) — MCP server for Claude, Cursor, Windsurf
- [x402 manifest](https://agentoracle.co/.well-known/x402.json) — agent-native pricing discovery

---

Built by [TK Collective](https://agentoracle.co) · x402 native · Base · SKALE · Stellar
