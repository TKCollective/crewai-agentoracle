"""AgentOracle tools for CrewAI agents."""

from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import requests


class VerifyInput(BaseModel):
    """Input schema for AgentOracle verify tool."""
    claim: str = Field(..., description="The claim or statement to verify for accuracy")


class EvaluateInput(BaseModel):
    """Input schema for AgentOracle evaluate tool."""
    content: str = Field(..., description="Content containing claims to evaluate")


class AgentOracleVerifyTool(BaseTool):
    name: str = "AgentOracle Verify"
    description: str = (
        "Verify whether a claim is true using AgentOracle's multi-source verification. "
        "Returns confidence score and recommendation. Free tier, 10 requests/hour."
    )
    args_schema: Type[BaseModel] = VerifyInput
    base_url: str = "https://agentoracle.co"

    def _run(self, claim: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/preview",
                json={"query": claim, "source": "crewai-agent"},
                headers={"Content-Type": "application/json"},
                timeout=30,
            )
            if response.status_code == 200:
                data = response.json()
                return (
                    f"Confidence: {data.get('confidence', 'N/A')}\n"
                    f"Summary: {data.get('summary', 'N/A')}\n"
                    f"Sources: {data.get('source_count', 'N/A')}"
                )
            return f"Verification returned status {response.status_code}"
        except Exception as e:
            return f"Verification failed: {str(e)}"


class AgentOracleEvaluateTool(BaseTool):
    name: str = "AgentOracle Evaluate"
    description: str = (
        "Full multi-source claim evaluation. Decomposes content into individual claims, "
        "verifies each against 4 independent sources (Sonar, Sonar Pro, Adversarial, Gemma 4), "
        "returns per-claim verdicts with confidence scores. Costs $0.02 USDC via x402."
    )
    args_schema: Type[BaseModel] = EvaluateInput
    base_url: str = "https://agentoracle.co"

    def _run(self, content: str) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/evaluate",
                json={"content": content, "source": "crewai-agent"},
                headers={"Content-Type": "application/json"},
                timeout=60,
            )
            if response.status_code == 200:
                data = response.json()
                evaluation = data.get("evaluation", {})
                claims = evaluation.get("claims", [])
                lines = [
                    f"Overall confidence: {evaluation.get('overall_confidence', 'N/A')}",
                    f"Recommendation: {evaluation.get('recommendation', 'N/A')}",
                    f"Claims: {evaluation.get('total_claims', 0)} total, "
                    f"{evaluation.get('verified_claims', 0)} verified, "
                    f"{evaluation.get('refuted_claims', 0)} refuted",
                ]
                for c in claims:
                    lines.append(f"- [{c.get('verdict','?')}] (conf:{c.get('confidence','?')}) {c.get('claim','?')[:100]}")
                return "\n".join(lines)
            elif response.status_code == 402:
                return "Payment required: $0.02 USDC via x402 on Base."
            return f"Evaluation returned status {response.status_code}"
        except Exception as e:
            return f"Evaluation failed: {str(e)}"
