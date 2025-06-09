from agents.agent_base import AgentBase
from ledger.mock_ledger import ledger_instance

agent_b = AgentBase("AgentB", {"translate_text": 3}, ledger_instance)