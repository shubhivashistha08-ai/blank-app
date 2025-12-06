# app.py
import streamlit as st
from agent import build_agent

st.set_page_config(page_title="FMCG Amazon Promo Agent", layout="wide")
st.markdown("<h2 style='text-align:center;'>FMCG Amazon Promotion Agent (Peanut Butter)</h2>",
            unsafe_allow_html=True)

st.write(
    "Chat with an agent that fetches live Amazon data for viral peanut/nut butter "
    "products via SerpAPI, evaluates promotions (discounts, sponsored positions) "
    "and suggests which strategies might work best."
)

# Build agent once
if "agent" not in st.session_state:
    st.session_state.agent = build_agent()
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if prompt := st.chat_input("Ask about campaigns, products, or promo strategies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Agent is analyzing live Amazon data..."):
            try:
                result = st.session_state.agent.invoke({"input": prompt})
                # result is an AIMessage; get its text
                answer = getattr(result, "content", str(result))
            except Exception as e:
                answer = f"Error from agent: {e}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
