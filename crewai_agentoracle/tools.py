"""
AgentOracle CrewAI Integration
Production-grade tools for per-claim trust verification in CrewAI agents.
All endpoints, full error handling, 402 payment support, retry logic.
"""
import time
import requests
from typing import Optional, Any, Dict, List, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool

AGENTORACLE_BASE_URL = "https://agentoracle.co"
DEFAULT_TIMEOUT = 60
MAX_RETRIES = 3
RETRY_BACKOFF = 2


def _make_request(
    endpoint: str,
    payload: Dict[str, Any],
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = MAX_RETRIES,
) -> Dict[str, Any]:
    """
    Core request handler with retry logic and 402 payment awareness.
    Returns structured dict — never raises on expected errors.
    """
    url = f"{AGENTORACLE_BASE_URL}{endpoint}"
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=timeout,
                headers={"Content-Type": "application/json"},
            )
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            if response.status_code == 402:
                return {
                    "success": False,
                    "error": "payment_required",
                    "message": (
                        "This endpoint requires x402 payment (USDC on Base). "
                        "Use /preview for free access (10 req/hr), or configure "
                        "an x402 wallet to access paid endpoints. "
                        "See: agentoracle.co/.well-known/x402.json"
                    ),
                    "payment_info": response.json() if response.text else {},
                }
            if response.status_code == 429:
                retry_after = int(response.headers.get("X-RateLimit-Reset", 60))
                return {
                    "success": False,
                    "error": "rate_limited",
                    "message": f"Rate limit exceeded. Resets in {retry_after} seconds.",
                    "retry_after": retry_after,
                }
            if response.status_code == 500:
                if attempt < retries - 1:
                    time.sleep(RETRY_BACKOFF ** attempt)
                    continue
                return {
                    "success": False,
                    "error": "server_error",
                    "message": "AgentOracle server error. Retry with exponential backoff.",
                }
            return {
                "success": False,
                "error": f"http_{response.status_code}",
                "message": response.text[:500],
            }
        except requests.Timeout:
            last_error = "Request timed out"
            if attempt < retries - 1:
                time.sleep(RETRY_BACKOFF ** attempt)
                continue
        except requests.ConnectionError:
            last_error = "Connection failed — check network or agentoracle.co status"
            if attempt < retries - 1:
                time.sleep(RETRY_BACKOFF ** attempt)
                continue
        except Exception as e:
            return {"success": False, "error": "unexpected", "message": str(e)}
    return {"success": False, "error": "max_retries", "message": last_error}


def _format_evaluation(data: Dict[str, Any]) -> str:
    """Format /evaluate response for CrewAI agent consumption."""
    ev = data.get("evaluation", {})
    if not ev:
        return f"Evaluation error: {data}"
    recommendation = ev.get("recommendation", "unknown").upper()
    overall = ev.get("overall_confidence", 0)
    lines = [
        f"=== AGENTORACLE EVALUATION ===",
        f"Recommendation: {recommendation}",
        f"Overall confidence: {overall:.2f}",
        f"Claims: {ev.get('total_claims', 0)} found | "
        f"{ev.get('verified_claims', 0)} supported | "
        f"{ev.get('refuted_claims', 0)} refuted | "
        f"{ev.get('unverifiable_claims', 0)} unverifiable",
        f"Sources: {', '.join(ev.get('sources_used', []))}",
        "",
        "=== PER-CLAIM VERDICTS ===",
    ]
    for claim in ev.get("claims", []):
        verdict = claim.get("verdict", "unknown").upper()
        confidence = claim.get("confidence", 0)
        text = claim.get("claim", "")
        evidence = claim.get("evidence", "")
        correction = claim.get("correction", "")
        counter = claim.get("counter_evidence", "")
        adversarial = claim.get("adversarial_result", "")
        symbol = {"SUPPORTED": "✓ PASS", "REFUTED": "✗ FAIL", "UNVERIFIABLE": "? UNKNOWN"}.get(
            verdict, "? UNKNOWN"
        )
        lines.append(f"\n{symbol} ({confidence:.2f})")
        lines.append(f"Claim: {text}")
        if evidence:
            lines.append(f"Evidence: {evidence[:300]}")
        if counter:
            lines.append(f"Counter-evidence: {counter[:200]}")
        if correction:
            lines.append(f"Correction: {correction}")
        if adversarial:
            lines.append(f"Adversarial scan: {adversarial}")
    gemma = data.get("gemma_calibration", {})
    if gemma:
        lines.append(
            f"\n=== GEMMA CALIBRATION ===\n"
            f"Calibrated confidence: {gemma.get('calibrated_confidence', 0):.2f} | "
            f"Agreement: {gemma.get('agreement', 'unknown')} | "
            f"Final recommendation: {gemma.get('recommendation', 'verify').upper()}"
        )
    meta = data.get("meta", {})
    lines.append(
        f"\nEvaluation ID: {data.get('evaluation_id', 'unknown')} | "
        f"Time: {meta.get('evaluation_time_ms', 0)}ms | "
        f"Cost: {meta.get('price', '$0.01 USDC')}"
    )
    lines.append(
        f"\n=== AGENT GUIDANCE ===\n"
        + {
            "ACT": "Confidence is high. Proceed with this information.",
            "VERIFY": "Confidence is moderate. Consider additional verification before acting.",
            "REJECT": "Confidence is low or claims are refuted. Do not act on this information.",
        }.get(recommendation, "Review claims before proceeding.")
    )
    return "\n".join(lines)


def _format_research(data: Dict[str, Any], query: str = "") -> str:
    """Format research response for CrewAI agent consumption."""
    lines = ["=== AGENTORACLE RESEARCH ==="]
    if query:
        lines.append(f"Query: {query}\n")
    lines.append(f"Summary: {data.get('summary', 'No summary available')}\n")
    facts = data.get("key_facts", [])
    if facts:
        lines.append("Key facts:")
        for fact in facts:
            lines.append(f"  • {fact}")
        lines.append("")
    sources = data.get("sources", [])
    if sources:
        lines.append("Sources:")
        for s in sources[:8]:
            url = s.get("url", s) if isinstance(s, dict) else s
            lines.append(f"  • {url}")
        lines.append("")
    confidence = data.get("confidence_score")
    if confidence is None:
        conf_obj = data.get("confidence", {})
        confidence = conf_obj.get("score") if isinstance(conf_obj, dict) else None
    if confidence is not None:
        lines.append(f"Confidence: {confidence:.2f}")
    meta = data.get("query_metadata", {})
    if meta:
        lines.append(
            f"Model: {meta.get('model', 'sonar')} | "
            f"Latency: {meta.get('latency_ms', 0)}ms | "
            f"Cost: ${meta.get('cost_usd', 0.02):.2f} USDC"
        )
    return "\n".join(lines)


# ─────────────────────────────────────────────
# INPUT SCHEMAS
# ─────────────────────────────────────────────
class EvaluateInput(BaseModel):
    content: str = Field(
        description=(
            "Text containing claims to verify. Can be raw text, research output, "
            "news excerpts, or any data retrieved by another agent. "
            "AgentOracle decomposes it into individual claims and verifies each one "
            "across 4 independent sources."
        )
    )
    source: Optional[str] = Field(
        default="crewai",
        description="Where this content came from (e.g. 'exa', 'perplexity', 'web', 'crewai').",
    )
    min_confidence: Optional[float] = Field(
        default=0.8,
        description="Confidence threshold for ACT recommendation (0.0-1.0).",
    )


class PreviewInput(BaseModel):
    query: str = Field(
        description="Research query for the free preview endpoint (10 req/hr, truncated results).",
    )


class ResearchInput(BaseModel):
    query: str = Field(description="Research query for full web research with sources and confidence score.")
    tier: Optional[str] = Field(
        default="standard",
        description="'standard' for Sonar ($0.02) or 'deep' for Sonar Pro ($0.10).",
    )


class DeepResearchInput(BaseModel):
    query: str = Field(
        description="Complex research query for multi-step deep analysis via Sonar Pro.",
    )


class BatchResearchInput(BaseModel):
    queries: List[str] = Field(
        description="List of research queries (max 10). $0.02 USDC each."
    )


class VerifyGateInput(BaseModel):
    content: str = Field(
        description=(
            "Text to run through the free pass/fail gate. "
            "Returns PASS or FAIL based on confidence threshold. "
            "No payment required. No per-claim breakdown."
        )
    )
    threshold: Optional[float] = Field(
        default=0.8,
        description="Confidence threshold for PASS decision (0.0-1.0).",
    )


# ─────────────────────────────────────────────
# TOOLS
# ─────────────────────────────────────────────
class AgentOracleVerifyTool(BaseTool):
    """
    Full 4-source claim verification via AgentOracle /evaluate.
    Core trust verification tool for CrewAI agents.
    $0.01 USDC per evaluation via x402 on Base.
    """

    name: str = "agentoracle_verify"
    description: str = (
        "Verify the trustworthiness of any content before your agent acts on it. "
        "Submits text to AgentOracle's 4-source verification pipeline: "
        "Sonar (real-time web), Sonar Pro (deep analysis), "
        "Adversarial (actively tries to disprove claims), and Gemma 4 (calibration). "
        "Returns per-claim verdicts (supported/refuted/unverifiable), "
        "confidence scores (0.00-1.00), and ACT/VERIFY/REJECT recommendation. "
        "Cost: $0.01 USDC via x402 on Base. "
        "Always use this before acting on data retrieved from external sources."
    )
    args_schema: Type[BaseModel] = EvaluateInput

    def _run(
        self,
        content: str,
        source: str = "crewai",
        min_confidence: float = 0.8,
    ) -> str:
        result = _make_request(
            "/evaluate",
            {
                "content": content,
                "source": source,
                "min_confidence": min_confidence,
            },
        )
        if not result["success"]:
            error = result.get("error", "unknown")
            message = result.get("message", "Unknown error")
            if error == "payment_required":
                return (
                    f"Payment required for /evaluate endpoint.\n"
                    f"{message}\n\n"
                    f"Fallback: Using free verify gate instead...\n"
                    + self._free_fallback(content)
                )
            return f"Verification failed ({error}): {message}"
        return _format_evaluation(result["data"])

    def _free_fallback(self, content: str) -> str:
        """Fallback to free verify-gate when payment not configured."""
        result = _make_request("/verify-gate", {"content": content, "threshold": 0.8})
        if not result["success"]:
            return "Free fallback also failed. Check agentoracle.co/health for status."
        data = result["data"]
        passed = data.get("passed", False)
        return (
            f"FREE GATE RESULT: {'PASS' if passed else 'FAIL'}\n"
            f"Confidence: {data.get('confidence', 0):.2f}\n"
            f"Configure x402 wallet for full per-claim analysis."
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use _run — async not yet supported")


class AgentOracleResearchTool(BaseTool):
    """
    Real-time web research via AgentOracle /research.
    $0.02 USDC per query via x402. Returns structured JSON with sources.
    """

    name: str = "agentoracle_research"
    description: str = (
        "Real-time web research with structured output. "
        "Returns summary, key facts, source URLs, and confidence score. "
        "Powered by Perplexity Sonar. "
        "Cost: $0.02 USDC via x402 on Base. "
        "For deeper analysis pass tier='deep' for Sonar Pro ($0.10). "
        "For claim verification use agentoracle_verify ($0.01/claim)."
    )
    args_schema: Type[BaseModel] = ResearchInput

    def _run(self, query: str, tier: str = "standard") -> str:
        payload: Dict[str, Any] = {"query": query}
        if tier == "deep":
            payload["tier"] = "deep"
        result = _make_request("/research", payload)
        if not result["success"]:
            error = result.get("error", "unknown")
            if error == "payment_required":
                return (
                    f"Payment required for /research.\n"
                    f"Using free preview instead (truncated)...\n"
                    + self._preview_fallback(query)
                )
            return f"Research failed ({error}): {result.get('message', 'Unknown error')}"
        return _format_research(result["data"], query)

    def _preview_fallback(self, query: str) -> str:
        result = _make_request("/preview", {"query": query})
        if not result["success"]:
            return "Preview fallback also failed."
        data = result["data"]
        return (
            f"PREVIEW (truncated): {data.get('summary', data.get('result', 'No summary'))}\n"
            f"Configure x402 wallet for full research results."
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use _run — async not yet supported")


class AgentOracleDeepResearchTool(BaseTool):
    """
    Deep multi-step research via AgentOracle /deep-research.
    $0.10 USDC per query. Uses Sonar Pro for comprehensive analysis.
    """

    name: str = "agentoracle_deep_research"
    description: str = (
        "Comprehensive multi-step research via Sonar Pro. "
        "Best for due diligence, market analysis, or complex questions. "
        "Returns extended analysis with higher confidence scoring. "
        "Cost: $0.10 USDC via x402 on Base. "
        "Use when standard research isn't sufficient."
    )
    args_schema: Type[BaseModel] = DeepResearchInput

    def _run(self, query: str) -> str:
        result = _make_request("/deep-research", {"query": query}, timeout=120)
        if not result["success"]:
            return f"Deep research failed: {result.get('message', 'Unknown error')}"
        return _format_research(result["data"], query)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use _run — async not yet supported")


class AgentOracleBatchResearchTool(BaseTool):
    """
    Batch research via AgentOracle /research/batch.
    $0.02 USDC per query. Efficient for multiple queries.
    """

    name: str = "agentoracle_batch_research"
    description: str = (
        "Run multiple research queries in one batch call. "
        "More efficient than individual calls for 3+ queries. "
        "Cost: $0.02 USDC per query via x402. Max 10 queries per batch."
    )
    args_schema: Type[BaseModel] = BatchResearchInput

    def _run(self, queries: List[str]) -> str:
        if not queries:
            return "No queries provided."
        if len(queries) > 10:
            return "Maximum 10 queries per batch."
        result = _make_request(
            "/research/batch",
            {"queries": queries},
            timeout=180,
        )
        if not result["success"]:
            return f"Batch research failed: {result.get('message', 'Unknown error')}"
        data = result["data"]
        results = data.get("results", [])
        if not results:
            return f"No results returned: {data}"
        lines = [f"=== BATCH RESEARCH ({len(results)} queries) ===\n"]
        for i, r in enumerate(results):
            q = queries[i] if i < len(queries) else f"Query {i+1}"
            lines.append(f"--- {q} ---")
            lines.append(_format_research(r, q))
            lines.append("")
        total = len(queries) * 0.02
        lines.append(f"Total cost: ${total:.2f} USDC")
        return "\n".join(lines)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use _run — async not yet supported")


class AgentOraclePreviewTool(BaseTool):
    """
    Free preview research via AgentOracle /preview.
    10 req/hr. No payment required. Truncated results.
    """

    name: str = "agentoracle_preview"
    description: str = (
        "Free research preview — no payment required. "
        "10 requests per hour. Returns truncated results with confidence score. "
        "Use for testing or when x402 payment is not configured. "
        "For full results use agentoracle_research ($0.02/query)."
    )
    args_schema: Type[BaseModel] = PreviewInput

    def _run(self, query: str) -> str:
        result = _make_request("/preview", {"query": query})
        if not result["success"]:
            return f"Preview failed: {result.get('message', 'Unknown error')}"
        data = result["data"]
        return (
            f"=== PREVIEW RESULT (truncated) ===\n"
            f"Query: {query}\n"
            f"Summary: {data.get('summary', data.get('result', 'No summary'))}\n"
            f"Confidence: {data.get('confidence_score', 'N/A')}\n"
            f"Note: Use agentoracle_research for full results and sources."
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use _run — async not yet supported")


class AgentOracleVerifyGateTool(BaseTool):
    """
    Free pass/fail verification gate via AgentOracle /verify-gate.
    No payment required. Simple boolean trust decision.
    """

    name: str = "agentoracle_verify_gate"
    description: str = (
        "Free pass/fail trust gate — no payment required. "
        "Quickly check if content meets a confidence threshold. "
        "Returns PASS or FAIL. No per-claim breakdown. "
        "Use agentoracle_verify for full per-claim analysis ($0.01)."
    )
    args_schema: Type[BaseModel] = VerifyGateInput

    def _run(self, content: str, threshold: float = 0.8) -> str:
        result = _make_request(
            "/verify-gate",
            {"content": content, "threshold": threshold},
        )
        if not result["success"]:
            return f"Verify gate failed: {result.get('message', 'Unknown error')}"
        data = result["data"]
        passed = data.get("passed", data.get("verified", False))
        confidence = data.get("confidence", data.get("score", 0))
        recommendation = data.get("recommendation", "verify")
        return (
            f"=== VERIFY GATE ===\n"
            f"Result: {'PASS' if passed else 'FAIL'}\n"
            f"Confidence: {confidence:.2f}\n"
            f"Recommendation: {recommendation.upper()}\n"
            f"Threshold: {threshold}\n"
            + ("Agent can proceed." if passed else "Agent should not act on this information.")
        )

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Use _run — async not yet supported")


# ─────────────────────────────────────────────
# CONVENIENCE — get all tools at once
# ─────────────────────────────────────────────
def get_agentoracle_tools(
    include_paid: bool = True,
    include_free: bool = True,
) -> List[BaseTool]:
    """
    Return all AgentOracle tools for use with CrewAI agents.

    Args:
        include_paid: Include tools requiring x402 payment
        include_free: Include free tools (preview, verify_gate)

    Returns:
        List of CrewAI BaseTool instances

    Example:
        from crewai_agentoracle import get_agentoracle_tools
        from crewai import Agent

        tools = get_agentoracle_tools()
        verifier = Agent(
            role="Fact Verifier",
            goal="Verify every claim before it reaches the writer",
            backstory="You trust nothing until AgentOracle confirms it.",
            tools=tools,
        )
    """
    tools = []
    if include_free:
        tools.extend([
            AgentOraclePreviewTool(),
            AgentOracleVerifyGateTool(),
        ])
    if include_paid:
        tools.extend([
            AgentOracleVerifyTool(),
            AgentOracleResearchTool(),
            AgentOracleDeepResearchTool(),
            AgentOracleBatchResearchTool(),
        ])
    return tools
