import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from agents.agent_a import agent_a
from agents.agent_b import agent_b
from ai.rl_negotiation import RLNegotiator
from ai.bandit_selection import BanditSelector
from communication.message_schema import ServiceRequest
from ledger.mock_ledger import ledger_instance

# Agent lookup
agents = {
    "AgentA": agent_a,
    "AgentB": agent_b
}

st.set_page_config(page_title="A2A Interaction Simulator", layout="centered")
st.title("🤖 A2A Economy Interaction Simulator")

st.markdown("### Select Agents and Service")

sender = st.selectbox("Requester (Sender)", options=list(agents.keys()))
receiver = st.selectbox("Provider (Receiver)", options=[a for a in agents.keys() if a != sender])

receiver_agent = agents[receiver]
services = list(receiver_agent.services.keys())

if services:
    service = st.selectbox("Service Type", options=services)
else:
    st.warning("This agent doesn't offer any services.")
    service = None

if service and st.button("Submit Request"):
    st.markdown("## 🔄 Processing...")

    request = ServiceRequest(sender=sender, receiver=receiver, service_type=service)

    # Negotiation phase
    negotiator = RLNegotiator()
    action = negotiator.get_action("state")  # placeholder state

    if action == "reject":
        st.error("🤝 Negotiation failed.")
    else:
        result = receiver_agent.handle_request(request)
        st.success(f"✅ {result}")

        st.markdown("### 💰 Updated Balances")
        st.write({
            agent: ledger_instance.accounts[agent]
            for agent in agents
        })
