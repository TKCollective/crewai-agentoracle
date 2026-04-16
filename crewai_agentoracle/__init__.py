"""CrewAI integration for AgentOracle — trust verification for AI agents."""
from crewai_agentoracle.tools import (
    AgentOracleVerifyTool,
    AgentOraclePreviewTool,
    AgentOracleResearchTool,
    AgentOracleDeepResearchTool,
    AgentOracleBatchResearchTool,
    AgentOracleVerifyGateTool,
    get_agentoracle_tools,
)

__all__ = [
    "AgentOracleVerifyTool",
    "AgentOraclePreviewTool",
    "AgentOracleResearchTool",
    "AgentOracleDeepResearchTool",
    "AgentOracleBatchResearchTool",
    "AgentOracleVerifyGateTool",
    "get_agentoracle_tools",
]

__version__ = "0.2.0"
