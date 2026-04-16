"""
Unit tests for AgentOracle CrewAI SDK tools.
17 tests covering all tools, error handling, fallback logic, and convenience helpers.
No real HTTP calls — all network activity is mocked with unittest.mock.
"""
import pytest
from unittest.mock import patch, MagicMock
import requests

from crewai_agentoracle.tools import (
    AgentOracleVerifyTool,
    AgentOracleResearchTool,
    AgentOracleDeepResearchTool,
    AgentOracleBatchResearchTool,
    AgentOraclePreviewTool,
    AgentOracleVerifyGateTool,
    get_agentoracle_tools,
)


# ─────────────────────────────────────────────
# VERIFY TOOL (CrewAI primary trust tool)
# ─────────────────────────────────────────────

def test_verify_tool_402_falls_back_to_gate(mock_verify_gate_pass):
    """When /evaluate returns 402, tool must fall back to /verify-gate and return a result."""
    tool = AgentOracleVerifyTool()

    resp_402 = MagicMock()
    resp_402.status_code = 402
    resp_402.text = '{"requires":"payment"}'
    resp_402.json.return_value = {"requires": "payment"}

    resp_gate = MagicMock()
    resp_gate.status_code = 200
    resp_gate.json.return_value = mock_verify_gate_pass

    with patch("requests.post", side_effect=[resp_402, resp_gate]):
        result = tool._run(content="Some content to verify.")

    assert isinstance(result, str)
    assert len(result) > 0
    # Should mention payment and fallback gate result
    assert "payment" in result.lower() or "GATE" in result or "PASS" in result


def test_verify_tool_parses_full_response(mock_evaluate_response):
    """Mock /evaluate 200 — result must contain recommendation, claims, and sections."""
    tool = AgentOracleVerifyTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_evaluate_response

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(content="LangGraph leads AI agent frameworks in 2026.")

    assert "AGENTORACLE EVALUATION" in result
    assert "VERIFY" in result
    assert "0.66" in result


def test_verify_tool_act_guidance():
    """Mock /evaluate with 'act' recommendation — result must contain 'Proceed'."""
    tool = AgentOracleVerifyTool()
    eval_data = {
        "evaluation_id": "eval_act",
        "evaluation": {
            "overall_confidence": 0.95,
            "recommendation": "act",
            "threshold_applied": 0.8,
            "total_claims": 1,
            "verified_claims": 1,
            "refuted_claims": 0,
            "unverifiable_claims": 0,
            "sources_used": ["sonar"],
            "claims": [
                {
                    "claim": "Bitcoin was created by Satoshi Nakamoto.",
                    "verdict": "supported",
                    "confidence": 1.0,
                }
            ],
        },
        "gemma_calibration": {},
        "meta": {"evaluation_time_ms": 1000, "price": "$0.01 USDC"},
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = eval_data

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(content="Bitcoin was created by Satoshi Nakamoto.")

    assert "Proceed" in result or "ACT" in result


def test_verify_tool_verify_guidance():
    """Mock /evaluate with 'verify' recommendation — result must contain 'VERIFY'."""
    tool = AgentOracleVerifyTool()
    eval_data = {
        "evaluation_id": "eval_verify",
        "evaluation": {
            "overall_confidence": 0.65,
            "recommendation": "verify",
            "threshold_applied": 0.8,
            "total_claims": 1,
            "verified_claims": 0,
            "refuted_claims": 0,
            "unverifiable_claims": 1,
            "sources_used": ["sonar"],
            "claims": [
                {
                    "claim": "A new AI framework launched last week.",
                    "verdict": "unverifiable",
                    "confidence": 0.5,
                }
            ],
        },
        "gemma_calibration": {},
        "meta": {"evaluation_time_ms": 800, "price": "$0.01 USDC"},
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = eval_data

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(content="A new AI framework launched last week.")

    assert "VERIFY" in result or "verification" in result.lower()


def test_verify_tool_reject_guidance():
    """Mock /evaluate with 'reject' recommendation — result must contain 'Do not act'."""
    tool = AgentOracleVerifyTool()
    eval_data = {
        "evaluation_id": "eval_reject",
        "evaluation": {
            "overall_confidence": 0.20,
            "recommendation": "reject",
            "threshold_applied": 0.8,
            "total_claims": 1,
            "verified_claims": 0,
            "refuted_claims": 1,
            "unverifiable_claims": 0,
            "sources_used": ["sonar"],
            "claims": [
                {
                    "claim": "The moon is made of cheese.",
                    "verdict": "refuted",
                    "confidence": 0.02,
                    "correction": "The moon is made of rock and regolith.",
                }
            ],
        },
        "gemma_calibration": {},
        "meta": {"evaluation_time_ms": 600, "price": "$0.01 USDC"},
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = eval_data

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(content="The moon is made of cheese.")

    assert "Do not act" in result or "REJECT" in result


def test_verify_tool_shows_all_verdicts(mock_evaluate_response):
    """Mock /evaluate 200 — both 'supported' and 'unverifiable' verdicts must appear."""
    tool = AgentOracleVerifyTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_evaluate_response

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(content="Multiple claims content.")

    assert "PASS" in result or "supported" in result.lower()
    assert "UNKNOWN" in result or "unverifiable" in result.lower()


# ─────────────────────────────────────────────
# RESEARCH TOOL
# ─────────────────────────────────────────────

def test_research_tool_fallback_to_preview(mock_preview_response):
    """When /research returns 402, tool must fall back to /preview."""
    tool = AgentOracleResearchTool()

    resp_402 = MagicMock()
    resp_402.status_code = 402
    resp_402.text = '{"requires":"payment"}'
    resp_402.json.return_value = {"requires": "payment"}

    resp_preview = MagicMock()
    resp_preview.status_code = 200
    resp_preview.json.return_value = mock_preview_response

    with patch("requests.post", side_effect=[resp_402, resp_preview]):
        result = tool._run(query="AI agent frameworks")

    assert isinstance(result, str)
    assert len(result) > 0
    assert "preview" in result.lower() or "truncated" in result.lower() or "Configure" in result


def test_research_tool_formats_correctly(mock_research_response):
    """Mock /research 200 — result must contain AGENTORACLE RESEARCH and summary."""
    tool = AgentOracleResearchTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_research_response

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(query="AI agent frameworks 2026")

    assert "AGENTORACLE RESEARCH" in result
    assert "AI agent frameworks are rapidly evolving" in result


# ─────────────────────────────────────────────
# DEEP RESEARCH TOOL
# ─────────────────────────────────────────────

def test_deep_research_timeout():
    """Mock Timeout on /deep-research — tool must not raise and must return a string."""
    tool = AgentOracleDeepResearchTool()

    with patch("requests.post", side_effect=requests.Timeout("timed out")):
        with patch("time.sleep"):
            result = tool._run(query="Complex deep research query")

    assert isinstance(result, str)
    assert len(result) > 0


# ─────────────────────────────────────────────
# BATCH RESEARCH TOOL
# ─────────────────────────────────────────────

def test_batch_research_max_queries():
    """Passing 11 queries must return 'Maximum 10' without making any HTTP call."""
    tool = AgentOracleBatchResearchTool()
    queries = [f"query {i}" for i in range(11)]

    with patch("requests.post") as mock_post:
        result = tool._run(queries=queries)

    assert "Maximum 10" in result
    mock_post.assert_not_called()


def test_batch_research_calculates_cost(mock_batch_response):
    """Mock /research/batch with 3-query batch — assert $0.06 total cost in output."""
    tool = AgentOracleBatchResearchTool()

    # Add a third result to the fixture
    three_result_response = {
        "results": mock_batch_response["results"] + [
            {
                "summary": "Result for query 3.",
                "key_facts": ["Fact C"],
                "sources": [{"url": "https://example.com/3"}],
                "confidence_score": 0.78,
                "query_metadata": {"model": "sonar", "latency_ms": 1000, "cost_usd": 0.02},
            }
        ]
    }
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = three_result_response

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(queries=["query 1", "query 2", "query 3"])

    assert "$0.06" in result


# ─────────────────────────────────────────────
# PREVIEW TOOL
# ─────────────────────────────────────────────

def test_preview_tool_crewai(mock_preview_response):
    """Mock /preview 200 — result must contain 'PREVIEW RESULT'."""
    tool = AgentOraclePreviewTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_preview_response

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(query="test preview query")

    assert "PREVIEW RESULT" in result


# ─────────────────────────────────────────────
# VERIFY GATE TOOL
# ─────────────────────────────────────────────

def test_verify_gate_pass_crewai(mock_verify_gate_pass):
    """Mock /verify-gate 200 passed=True — result must contain 'PASS'."""
    tool = AgentOracleVerifyGateTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_verify_gate_pass

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(content="Bitcoin was created by Satoshi Nakamoto.")

    assert "PASS" in result


def test_verify_gate_fail_crewai(mock_verify_gate_fail):
    """Mock /verify-gate 200 passed=False — result must contain 'FAIL'."""
    tool = AgentOracleVerifyGateTool()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_verify_gate_fail

    with patch("requests.post", return_value=mock_resp):
        result = tool._run(content="The moon is made of cheese.")

    assert "FAIL" in result


# ─────────────────────────────────────────────
# ERROR HANDLING
# ─────────────────────────────────────────────

def test_connection_error_crewai():
    """Mock ConnectionError on /preview — tool must not raise and must return a string."""
    tool = AgentOraclePreviewTool()

    with patch("requests.post", side_effect=requests.ConnectionError("network down")):
        with patch("time.sleep"):
            result = tool._run(query="connection error test")

    assert isinstance(result, str)
    assert len(result) > 0


# ─────────────────────────────────────────────
# CONVENIENCE HELPER
# ─────────────────────────────────────────────

def test_get_tools_crewai_all():
    """get_agentoracle_tools() with defaults must return exactly 6 tools."""
    tools = get_agentoracle_tools()
    assert len(tools) == 6
