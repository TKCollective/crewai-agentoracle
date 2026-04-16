"""
Shared pytest fixtures for AgentOracle LangChain SDK tests.
Provides mock API responses matching the real agentoracle.co response shapes.
"""
import pytest


@pytest.fixture
def mock_evaluate_response():
    """Mock /evaluate 200 response matching real AgentOracle shape."""
    return {
        "evaluation_id": "eval_test123",
        "evaluation": {
            "overall_confidence": 0.66,
            "recommendation": "verify",
            "threshold_applied": 0.8,
            "total_claims": 3,
            "verified_claims": 2,
            "refuted_claims": 0,
            "unverifiable_claims": 1,
            "verification_method": "multi-source",
            "sources_used": ["sonar", "sonar-pro", "adversarial", "gemma-4"],
            "claims": [
                {
                    "claim": "LangGraph leads AI agent frameworks in 2026.",
                    "verdict": "supported",
                    "confidence": 0.98,
                    "evidence": "Multiple sources confirm.",
                    "adversarial_result": "resistant",
                },
                {
                    "claim": "CrewAI was acquired by Google.",
                    "verdict": "unverifiable",
                    "confidence": 0.25,
                    "counter_evidence": "No sources confirm.",
                },
                {
                    "claim": "Bitcoin was created by Satoshi Nakamoto.",
                    "verdict": "supported",
                    "confidence": 1.0,
                    "adversarial_result": "resistant",
                },
            ],
        },
        "gemma_verification": {"verdicts": []},
        "gemma_calibration": {
            "calibrated_confidence": 0.65,
            "agreement": "moderate",
            "recommendation": "verify",
        },
        "meta": {
            "evaluation_time_ms": 25277,
            "price": "$0.01 USDC",
            "cache_hit": False,
        },
    }


@pytest.fixture
def mock_research_response():
    """Mock /research 200 response."""
    return {
        "summary": "AI agent frameworks are rapidly evolving in 2026.",
        "key_facts": [
            "LangGraph is widely used for stateful agents.",
            "CrewAI focuses on multi-agent orchestration.",
        ],
        "sources": [
            {"url": "https://example.com/ai-agents"},
            {"url": "https://example.com/langchain"},
        ],
        "confidence_score": 0.87,
        "query_metadata": {
            "model": "sonar",
            "latency_ms": 1200,
            "cost_usd": 0.02,
        },
    }


@pytest.fixture
def mock_preview_response():
    """Mock /preview 200 response (free, truncated)."""
    return {
        "summary": "Preview summary of AI agent frameworks.",
        "confidence_score": 0.75,
        "result": "Truncated preview result.",
    }


@pytest.fixture
def mock_verify_gate_pass():
    """Mock /verify-gate 200 response — PASS."""
    return {
        "passed": True,
        "confidence": 0.92,
        "recommendation": "act",
        "threshold": 0.8,
    }


@pytest.fixture
def mock_verify_gate_fail():
    """Mock /verify-gate 200 response — FAIL."""
    return {
        "passed": False,
        "confidence": 0.41,
        "recommendation": "reject",
        "threshold": 0.8,
    }


@pytest.fixture
def mock_batch_response():
    """Mock /research/batch 200 response with 2 results."""
    return {
        "results": [
            {
                "summary": "Result for query 1.",
                "key_facts": ["Fact A"],
                "sources": [{"url": "https://example.com/1"}],
                "confidence_score": 0.85,
                "query_metadata": {"model": "sonar", "latency_ms": 900, "cost_usd": 0.02},
            },
            {
                "summary": "Result for query 2.",
                "key_facts": ["Fact B"],
                "sources": [{"url": "https://example.com/2"}],
                "confidence_score": 0.80,
                "query_metadata": {"model": "sonar", "latency_ms": 950, "cost_usd": 0.02},
            },
        ]
    }
