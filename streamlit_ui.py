import streamlit as st
from agents.agent_a import agent_a
from agents.agent_b import agent_b

st.title("A2A Economy Dashboard")

st.write("Agent A Services:", agent_a.list_services())
st.write("Agent B Services:", agent_b.list_services())