# app.py
import streamlit as st
from agent import build_agent

st.set_page_config(page_title="FMCG Promo Agent", layout="wide")
st.markdown("<h2 style='text-align:center;'>FMCG Promotion & Campaign Agent</h2>", unsafe_allow_html=True)

st.write(
    "Ask about peanut butter (or other FMCG) promotions, discounts, sponsored listings, "
    "and which strategy might work best. The agent can inspect the dataset and run simple simulations."
)

# Build agent once and keep in session
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask about campaigns, promotions, or products..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking like a retail promo agent..."):
            try:
                result = st.session_state.agent.invoke({"input": prompt})
                answer = result["output"]
            except Exception as e:
                answer = f"Error from agent: {e}"
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
