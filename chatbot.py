# chatbot.py
import streamlit as st
import os
from groq import Groq

# --- Groq client setup ---
client = Groq(api_key="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")

# --- Session state init for chat ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def chatbot_reply(user_msg: str, automaton_info=None) -> str:
    """
    Query Groq LLM with user message + automaton context (if provided).
    """
    system_prompt = (
        "You are a tutor chatbot specialized in Formal Languages and Automata Theory. "
        "Answer questions clearly and concisely with helpful explanations."
    )

    if automaton_info:
        system_prompt += f"\nHere is the current automaton context:\n{automaton_info}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",   # Groq LLM, you can change if needed
        messages=[
            {"role": "system", "content": system_prompt},
            *st.session_state.chat_history,
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content


def render_chatbot(automaton_info=None):
    st.markdown(
        """
        <style>
        .chat-expander {
            position: fixed !important;
            bottom: 20px;
            right: 20px;
            width: 320px;
            z-index: 9999;
        }
        .chat-expander > details > summary {
            list-style: none;
        }
        .chat-expander > details > summary::-webkit-details-marker {
            display:none;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    with st.container():
        st.markdown('<div class="chat-expander">', unsafe_allow_html=True)
        with st.expander("💬 Chat with Automata Bot", expanded=False):
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.write(f"**You:** {msg['content']}")
                else:
                    st.write(f"**Bot:** {msg['content']}")

            user_input = st.text_input("Ask something...", key="chat_input", label_visibility="collapsed")
            if st.button("Send", key="send_chat"):
                if user_input:
                    st.session_state.chat_history.append({"role": "user", "content": user_input})
                    bot_reply = chatbot_reply(user_input, automaton_info)
                    st.session_state.chat_history.append({"role": "assistant", "content": bot_reply})
                    st.rerun()   # ✅ use new API
        st.markdown('</div>', unsafe_allow_html=True)