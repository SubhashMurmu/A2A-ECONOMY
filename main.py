from agents.agent_a import agent_a
from agents.agent_b import agent_b
from communication.message_schema import ServiceRequest
from ai.rl_negotiation import RLNegotiator
from ai.bandit_selection import BanditSelector
from ledger.mock_ledger import ledger_instance

def simulate_interaction():
    print("=== A2A Economy Simulation ===")

    # Show initial balances
    print("Initial Balances:")
    for agent in ["AgentA", "AgentB"]:
        print(f"{agent}: {ledger_instance.accounts[agent]} tokens")

    # AgentB requests 'clean_data' from AgentA
    request = ServiceRequest(sender="AgentB", receiver="AgentA", service_type="clean_data")

    # Use RL negotiator to simulate decision
    negotiator = RLNegotiator()
    action = negotiator.get_action("initial_state")
    print(f"Negotiation outcome: {action}")

    if action == "reject":
        print("Negotiation rejected.")
        return

    # Handle the request and simulate payment
    result = agent_a.handle_request(request)
    print("Transaction result:", result)

    # Bandit selector chooses next best agent
    selector = BanditSelector(["AgentA", "AgentB"])
    chosen = selector.select()
    print("Bandit-selected next preferred agent:", chosen)

    # Show updated balances
    print("Updated Balances:")
    for agent in ["AgentA", "AgentB"]:
        print(f"{agent}: {ledger_instance.accounts[agent]} tokens")

if __name__ == "__main__":
    simulate_interaction()
