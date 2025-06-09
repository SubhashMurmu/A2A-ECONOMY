from agents.agent_base import AgentBase
from ledger.mock_ledger import ledger_instance

agent_a = AgentBase("AgentA", {"clean_data": 5}, ledger_instance)